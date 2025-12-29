import asyncio
import json
import os
from functools import partial
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from nyrag.config import Config
from nyrag.logger import get_logger
from nyrag.utils import (
    DEFAULT_EMBEDDING_MODEL,
    get_vespa_tls_config,
    is_vespa_cloud,
    make_vespa_client,
    resolve_vespa_cloud_mtls_paths,
    resolve_vespa_port,
)


DEFAULT_ENDPOINT = "http://localhost:8080"
DEFAULT_RANKING = "default"
DEFAULT_SUMMARY = "top_k_chunks"


class SearchRequest(BaseModel):
    query: str = Field(..., description="User query string")
    hits: int = Field(10, description="Number of Vespa hits to return")
    k: int = Field(3, description="Top-k chunks to keep per hit")
    ranking: Optional[str] = Field(
        None, description="Ranking profile to use (defaults to schema default)"
    )
    summary: Optional[str] = Field(
        None, description="Document summary to request (defaults to top_k_chunks)"
    )


def _resolve_mtls_paths(
    vespa_url: str, project_folder: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    cert_env = (os.getenv("VESPA_CLIENT_CERT") or "").strip() or None
    key_env = (os.getenv("VESPA_CLIENT_KEY") or "").strip() or None

    if not is_vespa_cloud(vespa_url):
        return cert_env, key_env

    if cert_env or key_env:
        if not (cert_env and key_env):
            raise RuntimeError(
                "Vespa Cloud requires both VESPA_CLIENT_CERT and VESPA_CLIENT_KEY."
            )
        return cert_env, key_env

    if not project_folder:
        raise RuntimeError(
            "Vespa Cloud mTLS credentials not found. "
            "Export VESPA_CLIENT_CERT and VESPA_CLIENT_KEY with the paths to these files."
        )

    cert_path, key_path = resolve_vespa_cloud_mtls_paths(project_folder)
    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)

    raise RuntimeError(
        "Vespa Cloud mTLS credentials not found at "
        f"{cert_path} and {key_path}. "
        "Export VESPA_CLIENT_CERT and VESPA_CLIENT_KEY with the paths to these files."
    )


def _load_settings() -> Dict[str, Any]:
    """Load schema, model, and Vespa connection from env or YAML config."""
    config_path = os.getenv("NYRAG_CONFIG")
    vespa_url = (os.getenv("VESPA_URL") or "").strip() or "http://localhost"
    vespa_port = resolve_vespa_port(vespa_url)

    if config_path and Path(config_path).exists():
        cfg = Config.from_yaml(config_path)
        rag_params = cfg.rag_params or {}
        llm_config = cfg.get_llm_config()
        return {
            "app_package_name": cfg.get_app_package_name(),
            "schema_name": cfg.get_schema_name(),
            "embedding_model": rag_params.get(
                "embedding_model", DEFAULT_EMBEDDING_MODEL
            ),
            "vespa_url": vespa_url,
            "vespa_port": vespa_port,
            "llm_base_url": llm_config.get("llm_base_url"),
            "llm_model": llm_config.get("llm_model"),
            "llm_api_key": llm_config.get("llm_api_key"),
        }

    return {
        "app_package_name": None,
        "schema_name": os.getenv("VESPA_SCHEMA", "nyragwebrag"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "vespa_url": vespa_url,
        "vespa_port": vespa_port,
        "llm_base_url": None,
        "llm_model": None,
        "llm_api_key": None,
    }


settings = _load_settings()
logger = get_logger("api")
app = FastAPI(title="nyrag API", version="0.1.0")
model = SentenceTransformer(settings["embedding_model"])

# Get mTLS credentials (with Vespa Cloud fallback)
_cert, _key = _resolve_mtls_paths(
    settings["vespa_url"], settings.get("app_package_name")
)
_, _, _ca, _verify = get_vespa_tls_config()

vespa_app = make_vespa_client(
    settings["vespa_url"],
    settings["vespa_port"],
    _cert,
    _key,
    _ca,
    _verify,
)

base_dir = Path(__file__).parent
templates = Jinja2Templates(directory=str(base_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")


def _deep_find_numeric_field(obj: Any, key: str) -> Optional[float]:
    if isinstance(obj, dict):
        if key in obj:
            value = obj.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return None
        for v in obj.values():
            found = _deep_find_numeric_field(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _deep_find_numeric_field(item, key)
            if found is not None:
                return found
    return None


@app.get("/stats")
async def stats() -> Dict[str, Any]:
    """Return simple corpus statistics from Vespa (documents and chunks)."""
    doc_count: Optional[int] = None
    chunk_count: Optional[int] = None

    try:
        res = vespa_app.query(
            body={"yql": "select * from sources * where true", "hits": 0},
            schema=settings["schema_name"],
        )
        total = res.json.get("root", {}).get("fields", {}).get("totalCount")
        if isinstance(total, int):
            doc_count = total
        elif isinstance(total, str) and total.isdigit():
            doc_count = int(total)
    except Exception as e:
        logger.warning(f"Failed to fetch Vespa doc count: {e}")

    try:
        # Requires schema field `chunk_count` (added in this repo); if absent, this will likely return null.
        yql = (
            "select * from sources * where true | "
            "all(group(1) each(output(count(), sum(chunk_count))))"
        )
        res = vespa_app.query(
            body={"yql": yql, "hits": 0},
            schema=settings["schema_name"],
        )
        sum_value = _deep_find_numeric_field(res.json, "sum(chunk_count)")
        if sum_value is None:
            sum_value = _deep_find_numeric_field(res.json, "sum(chunk_count())")
        if sum_value is not None:
            chunk_count = int(sum_value)
    except Exception as e:
        logger.warning(f"Failed to fetch Vespa chunk count: {e}")

    return {
        "schema": settings["schema_name"],
        "documents": doc_count,
        "chunks": chunk_count,
    }


@app.post("/search")
async def search(req: SearchRequest) -> Dict[str, Any]:
    """Query Vespa using YQL with a precomputed query embedding."""
    embedding = model.encode(req.query, convert_to_numpy=True).tolist()
    body = {
        "yql": "select * from sources * where userInput(@query)",
        "query": req.query,
        "hits": req.hits,
        "summary": req.summary or DEFAULT_SUMMARY,
        "ranking.profile": req.ranking or DEFAULT_RANKING,
        "input.query(embedding)": embedding,
        "input.query(k)": req.k,
    }
    vespa_response = vespa_app.query(body=body, schema=settings["schema_name"])

    status_code = getattr(vespa_response, "status_code", 200)
    if status_code >= 400:
        detail = getattr(vespa_response, "json", vespa_response)
        raise HTTPException(status_code=status_code, detail=detail)

    return vespa_response.json


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Previous conversation messages as list of {role, content} dicts",
    )
    hits: int = Field(5, description="Number of Vespa hits to retrieve")
    k: int = Field(3, description="Top-k chunks per hit to keep")
    query_k: int = Field(
        3,
        ge=0,
        description="Number of alternate search queries to generate with the LLM",
    )
    model: Optional[str] = Field(
        None, description="OpenRouter model id (optional, uses env default if set)"
    )


def _fetch_chunks(query: str, hits: int, k: int) -> List[Dict[str, Any]]:
    embedding = model.encode(query, convert_to_numpy=True).tolist()
    body = {
        "yql": "select * from sources * where userInput(@query)",
        "query": query,
        "hits": hits,
        "summary": DEFAULT_SUMMARY,
        "ranking.profile": DEFAULT_RANKING,
        "input.query(embedding)": embedding,
        "input.query(k)": k,
        "presentation.summaryFeatures": True,
    }
    vespa_response = vespa_app.query(body=body, schema=settings["schema_name"])
    status_code = getattr(vespa_response, "status_code", 200)
    if status_code >= 400:
        detail = getattr(vespa_response, "json", vespa_response)
        raise HTTPException(status_code=status_code, detail=detail)

    hits_data = vespa_response.json.get("root", {}).get("children", []) or []
    chunks: List[Dict[str, Any]] = []
    for hit in hits_data:
        fields = hit.get("fields", {}) or {}
        loc = fields.get("loc") or fields.get("id") or ""
        chunk_texts = fields.get("chunks_topk") or []
        hit_score_raw = hit.get("relevance", 0.0)
        logger.info(f"Hit loc={loc} score={hit_score_raw} chunks={len(chunk_texts)}")
        try:
            hit_score = float(hit_score_raw)
        except (TypeError, ValueError):
            hit_score = 0.0
        summary_features = (
            hit.get("summaryfeatures")
            or hit.get("summaryFeatures")
            or fields.get("summaryfeatures")
            or {}
        )
        chunk_score_raw = summary_features.get("best_chunk_score", hit_score)
        logger.info(f"  best_chunk_score={chunk_score_raw}")
        try:
            chunk_score = float(chunk_score_raw)
        except (TypeError, ValueError):
            chunk_score = hit_score

        for chunk in chunk_texts:
            chunks.append(
                {
                    "loc": loc,
                    "chunk": chunk,
                    "score": chunk_score,
                    "hit_score": hit_score,
                    "source_query": query,
                }
            )
    return chunks


async def _fetch_chunks_async(query: str, hits: int, k: int) -> List[Dict[str, Any]]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(_fetch_chunks, query, hits, k))


def _get_llm_client() -> AsyncOpenAI:
    """Get LLM client supporting any OpenAI-compatible API (OpenRouter, Ollama, LM Studio, vLLM, etc.)."""
    # Priority: config file > env vars > defaults (OpenRouter)
    base_url = (
        settings.get("llm_base_url")
        or os.getenv("LLM_BASE_URL")
        or os.getenv("OPENROUTER_BASE_URL")
        or "https://openrouter.ai/api/v1"
    )

    api_key = (
        settings.get("llm_api_key")
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    )

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="LLM API key not set. Set LLM_API_KEY or OPENROUTER_API_KEY environment variable, "
            "or configure llm_api_key in config file. For local models, use any dummy value.",
        )

    # OpenRouter-specific headers (ignored by other providers)
    default_headers = {}
    referer = os.getenv("OPENROUTER_REFERRER")
    if referer:
        default_headers["HTTP-Referer"] = referer
    title = os.getenv("OPENROUTER_TITLE")
    if title:
        default_headers["X-Title"] = title

    return AsyncOpenAI(
        base_url=base_url, api_key=api_key, default_headers=default_headers or None
    )


def _extract_message_text(content: Any) -> str:
    """Handle OpenAI response content that may be str or list of text blocks."""
    if content is None:
        return ""
    if isinstance(content, dict) and "text" in content:
        return str(content.get("text", ""))
    if hasattr(content, "text"):
        return str(getattr(content, "text", ""))
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")))
            elif hasattr(part, "text"):
                texts.append(str(getattr(part, "text", "")))
            elif isinstance(part, str):
                texts.append(part)
        return "\n".join([t for t in texts if t])
    return str(content)


async def _generate_search_queries_stream(
    user_message: str,
    model_id: str,
    num_queries: int,
    hits: int,
    k: int,
    history: Optional[List[Dict[str, str]]] = None,
) -> AsyncGenerator[Tuple[str, Any], None]:
    """Use the chat LLM to propose focused search queries grounded in retrieved chunks."""
    if num_queries <= 0:
        yield "result", []
        return

    grounding_chunks = (await _fetch_chunks_async(user_message, hits=hits, k=k))[:5]
    grounding_text = "\n".join(
        f"- [{c.get('loc','')}] {c.get('chunk','')}" for c in grounding_chunks
    )

    system_prompt = (
        "You generate concise, to-the-point search queries that help retrieve"
        " factual context for answering the user."
        " Do not change the meaning of the question."
        " Do not introduce any new information, words, concepts, or ideas."
        " Do not add any new words."
        " Prefer to reuse the provided context to stay on-topic."
        "Return only valid JSON."
    )

    # Build conversation context if history exists
    conversation_context = ""
    if history:
        conversation_context = "Previous conversation:\n"
        for msg in history[-4:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]
            conversation_context += f"{role}: {content}\n"
        conversation_context += "\n"

    user_prompt = (
        f"{conversation_context}"
        f"Create {num_queries} diverse, specific search queries (max 12 words each)"
        f' that would retrieve evidence to answer:\n"{user_message}".\n'
        f"Grounding context:\n{grounding_text or '(no context found)'}\n"
        'Respond as a JSON object like {"queries": ["query 1", "query 2"]}.'
    )

    full_content = ""
    try:
        client = _get_llm_client()

        # Try with response_format and reasoning support (OpenRouter, OpenAI)
        # Fall back to basic streaming if not supported (local models)
        request_kwargs = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
        }

        # Try with structured output first
        try:
            request_kwargs["response_format"] = {"type": "json_object"}
            request_kwargs["extra_body"] = {"reasoning": {"enabled": True}}
            stream = await client.chat.completions.create(**request_kwargs)
        except Exception as e:
            # Fallback: remove unsupported parameters for local models
            if (
                "response_format" in str(e)
                or "extra_body" in str(e)
                or "reasoning" in str(e)
            ):
                request_kwargs.pop("response_format", None)
                request_kwargs.pop("extra_body", None)
                stream = await client.chat.completions.create(**request_kwargs)
            else:
                raise

        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            reasoning = getattr(delta, "reasoning", None)
            reasoning_text = _extract_message_text(reasoning)
            if reasoning_text:
                yield "thinking", reasoning_text

            content_piece = _extract_message_text(getattr(delta, "content", None))
            if content_piece:
                full_content += content_piece
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    queries: List[str] = []
    try:
        parsed = json.loads(full_content)
        candidates = parsed.get("queries") if isinstance(parsed, dict) else parsed
        if isinstance(candidates, list):
            queries = [str(q).strip() for q in candidates if str(q).strip()]
    except Exception:
        queries = []

    # Fallback: try to parse line-separated text if JSON parsing fails
    if not queries:
        for line in full_content.splitlines():
            candidate = line.strip(" -•\t")
            if candidate:
                queries.append(candidate)

    cleaned: List[str] = []
    seen: Set[str] = set()
    for q in queries:
        q_norm = q.strip()
        key = q_norm.lower()
        if q_norm and key not in seen:
            cleaned.append(q_norm)
            seen.add(key)
        if len(cleaned) >= num_queries:
            break
    yield "result", cleaned


async def _prepare_queries_stream(
    user_message: str,
    model_id: str,
    query_k: int,
    hits: int,
    k: int,
    history: Optional[List[Dict[str, str]]] = None,
) -> AsyncGenerator[Tuple[str, Any], None]:
    """Build the list of queries (original + enhanced) for retrieval."""
    enhanced = []
    async for event_type, payload in _generate_search_queries_stream(
        user_message, model_id, query_k, hits=hits, k=k, history=history
    ):
        if event_type == "thinking":
            yield "thinking", payload
        elif event_type == "result":
            enhanced = payload

    queries = [user_message] + enhanced

    deduped: List[str] = []
    seen: Set[str] = set()
    for q in queries:
        q_norm = q.strip()
        key = q_norm.lower()
        if q_norm and key not in seen:
            deduped.append(q_norm)
            seen.add(key)
    logger.info(f"Search queries ({len(deduped)}): {deduped}")
    yield "result", deduped


async def _prepare_queries(
    user_message: str, model_id: str, query_k: int, hits: int, k: int
) -> List[str]:
    queries = []
    async for event_type, payload in _prepare_queries_stream(
        user_message, model_id, query_k, hits, k
    ):
        if event_type == "result":
            queries = payload
    return queries


async def _fuse_chunks(
    queries: List[str], hits: int, k: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Search Vespa for each query and return fused, deduped chunks."""
    all_chunks: List[Dict[str, Any]] = []
    logger.info(f"Fetching chunks for {len(queries)} queries")

    tasks = [_fetch_chunks_async(q, hits=hits, k=k) for q in queries]
    results = await asyncio.gather(*tasks)
    for res in results:
        all_chunks.extend(res)

    logger.info(f"Fetched total {len(all_chunks)} chunks from Vespa")
    if not all_chunks:
        return queries, []

    max_context = hits * k
    if max_context <= 0:
        max_context = len(all_chunks)

    # Aggregate duplicates (same loc+chunk) and average their scores.
    aggregated: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for chunk in all_chunks:
        key = (chunk.get("loc", ""), chunk.get("chunk", ""))
        score = float(chunk.get("score", chunk.get("hit_score", 0.0)) or 0.0)
        hit_score = float(chunk.get("hit_score", 0.0) or 0.0)
        source_query = chunk.get("source_query")

        if key not in aggregated:
            aggregated[key] = {
                "loc": key[0],
                "chunk": key[1],
                "score_sum": score,
                "hit_sum": hit_score,
                "count": 1,
                "source_queries": [source_query] if source_query else [],
            }
        else:
            agg = aggregated[key]
            agg["score_sum"] += score
            agg["hit_sum"] += hit_score
            agg["count"] += 1
            if source_query and source_query not in agg["source_queries"]:
                agg["source_queries"].append(source_query)

    fused: List[Dict[str, Any]] = []
    for agg in aggregated.values():
        count = max(agg.pop("count", 1), 1)
        agg["score"] = agg.pop("score_sum", 0.0) / count
        agg["hit_score"] = agg.pop("hit_sum", 0.0) / count
        sources = agg.get("source_queries") or []
        agg["source_query"] = sources[0] if sources else ""
        fused.append(agg)

    fused.sort(key=lambda c: c.get("score", c.get("hit_score", 0.0)), reverse=True)
    fused = fused[:max_context]

    return queries, fused


async def _call_openrouter(
    context: List[Dict[str, str]], user_message: str, model_id: str
) -> str:
    system_prompt = (
        "You are a helpful assistant. "
        "Answer user's question using only the provided context. "
        "Provide elaborate and informative answers where possible. "
        "If the context is insufficient, say you don't know."
    )
    context_text = "\n\n".join(
        [f"[{c.get('loc','')}] {c.get('chunk','')}" for c in context]
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Grounding context:\n{context_text}\n\nQuestion: {user_message}",
        },
    ]

    try:
        client = _get_llm_client()

        # Try with reasoning support, fall back without it
        try:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=messages,
                extra_body={"reasoning": {"enabled": True}},
            )
        except Exception as e:
            # Fallback: remove unsupported parameters for local models
            if "extra_body" in str(e) or "reasoning" in str(e):
                resp = await client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                )
            else:
                raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return _extract_message_text(resp.choices[0].message.content)


async def _openrouter_stream(
    context: List[Dict[str, str]],
    user_message: str,
    model_id: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> AsyncGenerator[Tuple[str, str], None]:
    system_prompt = (
        "You are a helpful assistant. Answer using only the provided context. "
        "If the context is insufficient, say you don't know."
    )
    context_text = "\n\n".join(
        [f"[{c.get('loc','')}] {c.get('chunk','')}" for c in context]
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history if provided
    if history:
        messages.extend(history)

    # Add current user message with context
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {user_message}",
        }
    )

    try:
        client = _get_llm_client()

        # Try with reasoning support, fall back without it
        try:
            stream = await client.chat.completions.create(
                model=model_id,
                messages=messages,
                stream=True,
                extra_body={"reasoning": {"enabled": True}},
            )
        except Exception as e:
            # Fallback: remove unsupported parameters for local models
            if "extra_body" in str(e) or "reasoning" in str(e):
                stream = await client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    stream=True,
                )
            else:
                raise

        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta
            reasoning = _extract_message_text(getattr(delta, "reasoning", None))
            if reasoning:
                yield "thinking", reasoning

            content_piece = _extract_message_text(getattr(delta, "content", None))
            if content_piece:
                yield "token", content_piece
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat")
async def chat(req: ChatRequest) -> Dict[str, Any]:
    model_id = (
        req.model
        or settings.get("llm_model")
        or os.getenv("LLM_MODEL")
        or os.getenv("OPENROUTER_MODEL")
    )
    queries, chunks = await _fuse_chunks(
        await _prepare_queries(
            req.message, model_id, req.query_k, hits=req.hits, k=req.k
        ),
        hits=req.hits,
        k=req.k,
    )
    if not chunks:
        return {"answer": "No relevant context found.", "chunks": []}
    answer = await _call_openrouter(chunks, req.message, model_id)
    return {"answer": answer, "chunks": chunks, "queries": queries}


@app.post("/chat-stream")
async def chat_stream(req: ChatRequest):
    model_id = (
        req.model
        or settings.get("llm_model")
        or os.getenv("LLM_MODEL")
        or os.getenv("OPENROUTER_MODEL")
    )

    async def event_stream():
        yield f"data: {json.dumps({'type': 'status', 'payload': 'Generating search queries...'})}\n\n"

        queries = []
        async for event_type, payload in _prepare_queries_stream(
            req.message,
            model_id,
            req.query_k,
            hits=req.hits,
            k=req.k,
            history=req.history,
        ):
            if event_type == "thinking":
                yield f"data: {json.dumps({'type': 'thinking', 'payload': payload})}\n\n"
            elif event_type == "result":
                queries = payload

        yield f"data: {json.dumps({'type': 'queries', 'payload': queries})}\n\n"
        yield f"data: {json.dumps({'type': 'status', 'payload': 'Retrieving context from Vespa...'})}\n\n"
        queries, chunks = await _fuse_chunks(queries, hits=req.hits, k=req.k)
        yield f"data: {json.dumps({'type': 'chunks', 'payload': chunks})}\n\n"
        if not chunks:
            yield f"data: {json.dumps({'type': 'done', 'payload': 'No relevant context found.'})}\n\n"
            return
        yield f"data: {json.dumps({'type': 'status', 'payload': 'Generating answer...'})}\n\n"
        async for type_, payload in _openrouter_stream(
            chunks, req.message, model_id, req.history
        ):
            yield f"data: {json.dumps({'type': type_, 'payload': payload})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("chat.html", {"request": request})
