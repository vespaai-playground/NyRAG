import asyncio
import json
import os
import sys
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from nyrag.config import Config, get_config_options, get_example_configs
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


def _normalize_project_name(name: str) -> str:
    clean_name = name.replace("-", "").replace("_", "").lower()
    return f"nyrag{clean_name}"


def _resolve_config_path(
    project_name: Optional[str] = None,
    config_yaml: Optional[str] = None,
    active_project: Optional[str] = None,
) -> Path:
    if project_name:
        return Path("output") / project_name / "conf.yml"

    if config_yaml is not None:
        import yaml

        try:
            config_data = yaml.safe_load(config_yaml) or {}
        except yaml.YAMLError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}") from exc
        raw_name = config_data.get("name") or "project"
        schema_name = _normalize_project_name(str(raw_name))
        return Path("output") / schema_name / "conf.yml"

    if active_project:
        return Path("output") / active_project / "conf.yml"
    raise HTTPException(status_code=400, detail="project_name is required")


class SearchRequest(BaseModel):
    query: str = Field(..., description="User query string")
    hits: int = Field(10, description="Number of Vespa hits to return")
    k: int = Field(3, description="Top-k chunks to keep per hit")
    ranking: Optional[str] = Field(None, description="Ranking profile to use (defaults to schema default)")
    summary: Optional[str] = Field(None, description="Document summary to request (defaults to top_k_chunks)")


class CrawlRequest(BaseModel):
    config_yaml: str = Field(..., description="YAML configuration content")


def _resolve_mtls_paths(vespa_url: str, project_folder: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    cert_env = (os.getenv("VESPA_CLIENT_CERT") or "").strip() or None
    key_env = (os.getenv("VESPA_CLIENT_KEY") or "").strip() or None

    if not is_vespa_cloud(vespa_url):
        return cert_env, key_env

    if cert_env or key_env:
        if not (cert_env and key_env):
            raise RuntimeError("Vespa Cloud requires both VESPA_CLIENT_CERT and VESPA_CLIENT_KEY.")
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
            # Env vars override config file
            "schema_name": os.getenv("VESPA_SCHEMA") or cfg.get_schema_name(),
            "embedding_model": os.getenv("EMBEDDING_MODEL")
            or rag_params.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
            "vespa_url": vespa_url,
            "vespa_port": vespa_port,
            "llm_base_url": os.getenv("LLM_BASE_URL") or llm_config.get("llm_base_url"),
            "llm_model": os.getenv("LLM_MODEL") or llm_config.get("llm_model"),
            "llm_api_key": os.getenv("LLM_API_KEY") or llm_config.get("llm_api_key"),
        }

    return {
        "app_package_name": None,
        "schema_name": os.getenv("VESPA_SCHEMA"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "vespa_url": vespa_url,
        "vespa_port": vespa_port,
        "llm_base_url": None,
        "llm_model": None,
        "llm_api_key": None,
    }


def list_available_projects() -> List[str]:
    """List available projects (folders with conf.yml)."""
    projects = []
    output_dir = Path("output")
    if output_dir.exists():
        for project_dir in sorted(output_dir.iterdir()):
            if project_dir.is_dir() and (project_dir / "conf.yml").exists():
                projects.append(project_dir.name)
    return projects


def load_project_settings(project_name: str) -> Dict[str, Any]:
    """Load settings from a specific project's conf.yml."""
    vespa_url = (os.getenv("VESPA_URL") or "").strip() or "http://localhost"
    vespa_port = resolve_vespa_port(vespa_url)

    config_path = Path("output") / project_name / "conf.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Project config not found: {config_path}")

    cfg = Config.from_yaml(str(config_path))
    rag_params = cfg.rag_params or {}
    llm_config = cfg.get_llm_config()

    return {
        "app_package_name": cfg.get_app_package_name(),
        # Env vars override config file
        "schema_name": os.getenv("VESPA_SCHEMA") or cfg.get_schema_name(),
        "embedding_model": os.getenv("EMBEDDING_MODEL") or rag_params.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        "vespa_url": vespa_url,
        "vespa_port": vespa_port,
        "llm_base_url": os.getenv("LLM_BASE_URL") or llm_config.get("llm_base_url"),
        "llm_model": os.getenv("LLM_MODEL") or llm_config.get("llm_model"),
        "llm_api_key": os.getenv("LLM_API_KEY") or llm_config.get("llm_api_key"),
    }


class CrawlManager:
    def __init__(self):
        self.process = None
        self.subscribers: List[asyncio.Queue] = []
        self.temp_config_path: Optional[str] = None

    async def start_crawl(self, config_yaml: str):
        if self.process and self.process.returncode is None:
            return  # Already running

        # Parse config to get output path and save conf.yml
        import yaml

        config_data = yaml.safe_load(config_yaml) or {}
        project_name = config_data.get("name", "project")
        schema_name = _normalize_project_name(str(project_name))
        output_dir = Path("output") / schema_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config to output folder
        config_path = output_dir / "conf.yml"
        with open(config_path, "w") as f:
            f.write(config_yaml)
        logger.info(f"Config saved to {config_path}")

        # Also create temp file for the subprocess
        fd, self.temp_config_path = tempfile.mkstemp(suffix=".yml", text=True)
        with os.fdopen(fd, "w") as f:
            f.write(config_yaml)

        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "nyrag.cli",
            "--config",
            self.temp_config_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        asyncio.create_task(self._read_logs())

    async def _read_logs(self):
        if not self.process:
            return

        while True:
            line = await self.process.stdout.readline()
            if not line:
                break
            decoded_line = line.decode("utf-8").rstrip()
            # Log to server terminal
            logger.info(decoded_line)
            for q in self.subscribers:
                await q.put(decoded_line)

        await self.process.wait()

        # Cleanup temp file
        if self.temp_config_path and os.path.exists(self.temp_config_path):
            try:
                os.unlink(self.temp_config_path)
            except OSError:
                pass
        self.temp_config_path = None

        # Notify completion
        for q in self.subscribers:
            await q.put("EOF")

    async def stop_crawl(self):
        """Stop the running crawl process."""
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()

            # Notify subscribers
            for q in self.subscribers:
                await q.put("EOF")

            return True
        return False

    async def stream_logs(self):
        q = asyncio.Queue()
        self.subscribers.append(q)
        try:
            while True:
                line = await q.get()
                if line == "EOF":
                    yield "data: [PROCESS COMPLETED]\n\n"
                    break
                yield f"data: {line}\n\n"
        finally:
            if q in self.subscribers:
                self.subscribers.remove(q)


crawl_manager = CrawlManager()

active_project: Optional[str] = None
settings = _load_settings()
logger = get_logger("api")
app = FastAPI(title="nyrag API", version="0.1.0")
model = SentenceTransformer(settings["embedding_model"])

# Get mTLS credentials (with Vespa Cloud fallback)
_cert, _key = _resolve_mtls_paths(settings["vespa_url"], settings.get("app_package_name"))
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


@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


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
        yql = "select * from sources * where true | " "all(group(1) each(output(count(), sum(chunk_count))))"
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


class ConfigContent(BaseModel):
    content: str


@app.get("/config/options")
async def get_config_schema(mode: str = "web") -> Dict[str, Any]:
    """Get the configuration schema options for the frontend."""
    return get_config_options(mode)


@app.get("/config")
async def get_config(project_name: Optional[str] = None) -> Dict[str, str]:
    """Get content of the project configuration file."""
    if not project_name and not active_project:
        return {"content": ""}
    config_path = _resolve_config_path(project_name=project_name, active_project=active_project)
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"Project config not found: {config_path}")

    with open(config_path, "r") as f:
        return {"content": f.read()}


@app.post("/config")
async def save_config(config: ConfigContent):
    """Save content to the project configuration file."""
    config_path = _resolve_config_path(config_yaml=config.content, active_project=active_project)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        f.write(config.content)
    global active_project
    active_project = config_path.parent.name

    return {"status": "saved"}


@app.get("/config/examples")
async def list_example_configs() -> Dict[str, str]:
    """List available example configurations."""
    return get_example_configs()


@app.get("/config/mode")
async def get_config_mode():
    """Check if NYRAG_CONFIG env var is set."""
    config_path = os.getenv("NYRAG_CONFIG")
    if config_path and Path(config_path).exists():
        return {"mode": "env_config", "config_path": config_path, "allow_project_selection": False}
    return {"mode": "project_selection", "config_path": None, "allow_project_selection": True}


@app.get("/projects")
async def get_projects():
    """List available projects."""
    config_path = os.getenv("NYRAG_CONFIG")
    # If NYRAG_CONFIG is set, don't list projects
    if config_path and Path(config_path).exists():
        return []
    return list_available_projects()


@app.post("/projects/select")
async def select_project(project_name: str = Body(..., embed=True)):
    """Select a project and load its settings."""
    # If NYRAG_CONFIG is set, don't allow project switching
    if os.getenv("NYRAG_CONFIG"):
        raise HTTPException(status_code=403, detail="Project selection disabled when NYRAG_CONFIG is set")

    global active_project, settings
    try:
        settings = load_project_settings(project_name)
        active_project = project_name
        return {"status": "success", "settings": settings}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/crawl/start")
async def start_crawl(req: CrawlRequest):
    await crawl_manager.start_crawl(req.config_yaml)
    return {"status": "started"}


@app.get("/crawl/logs")
async def stream_crawl_logs():
    return StreamingResponse(crawl_manager.stream_logs(), media_type="text/event-stream")


@app.post("/crawl/stop")
async def stop_crawl():
    """Stop the running crawl process."""
    stopped = await crawl_manager.stop_crawl()
    return {"status": "stopped" if stopped else "not_running"}


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
    model: Optional[str] = Field(None, description="OpenRouter model id (optional, uses env default if set)")


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
            hit.get("summaryfeatures") or hit.get("summaryFeatures") or fields.get("summaryfeatures") or {}
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
    # Priority: env vars > config file > defaults (OpenRouter)
    # Note: settings already has env vars applied with higher priority from _load_settings()
    base_url = settings.get("llm_base_url") or os.getenv("LLM_BASE_URL") or "https://openrouter.ai/api/v1"

    api_key = settings.get("llm_api_key") or os.getenv("LLM_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="LLM API key not set. Set LLM_API_KEY environment variable, "
            "or configure llm_api_key in config file. For local models, use any dummy value.",
        )

    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def _resolve_model_id(request_model: Optional[str]) -> str:
    # Priority: request param > settings (which has env > config) > env fallback
    # Note: settings already has LLM_MODEL env var applied with higher priority
    model_id = (
        (request_model or "").strip()
        or (settings.get("llm_model") or "").strip()
        or (os.getenv("LLM_MODEL") or "").strip()
    )
    if not model_id:
        raise HTTPException(
            status_code=500,
            detail="LLM model not set. Set LLM_MODEL env var, configure llm_config.model in the project config, "
            "or pass model in the request.",
        )
    return model_id


async def _create_chat_completion_with_fallback(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, str]],
    stream: bool = False,
    enable_json_mode: bool = False,
    enable_reasoning: bool = False,
) -> Any:
    """
    Create a chat completion with graceful fallback for unsupported features.

    Tries advanced features first (json_object, reasoning), then falls back to basic mode
    if the server doesn't support them (e.g., local models like Ollama, LM Studio).

    Args:
        client: AsyncOpenAI client
        model: Model name
        messages: List of message dictionaries
        stream: Whether to stream responses
        enable_json_mode: Whether to request JSON output format
        enable_reasoning: Whether to enable reasoning mode

    Returns:
        Chat completion response or stream
    """
    request_kwargs = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }

    # Add optional features
    if enable_json_mode:
        request_kwargs["response_format"] = {"type": "json_object"}
    if enable_reasoning:
        request_kwargs["extra_body"] = {"reasoning": {"enabled": True}}

    # Try with all features first
    if enable_json_mode or enable_reasoning:
        try:
            return await client.chat.completions.create(**request_kwargs)
        except Exception as e:
            # Check if error is related to unsupported features
            error_str = str(e).lower()
            if any(
                keyword in error_str
                for keyword in [
                    "response_format",
                    "extra_body",
                    "reasoning",
                    "json_object",
                ]
            ):
                # Fallback: remove unsupported parameters
                request_kwargs.pop("response_format", None)
                request_kwargs.pop("extra_body", None)
                return await client.chat.completions.create(**request_kwargs)
            else:
                # Different error, re-raise
                raise

    # No special features requested, just make the call
    return await client.chat.completions.create(**request_kwargs)


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
    grounding_text = "\n".join(f"- [{c.get('loc','')}] {c.get('chunk','')}" for c in grounding_chunks)

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
        stream = await _create_chat_completion_with_fallback(
            client=client,
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            enable_json_mode=True,
            enable_reasoning=True,
        )

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


async def _prepare_queries(user_message: str, model_id: str, query_k: int, hits: int, k: int) -> List[str]:
    model_id = _resolve_model_id(model_id)
    queries = []
    async for event_type, payload in _prepare_queries_stream(user_message, model_id, query_k, hits, k):
        if event_type == "result":
            queries = payload
    return queries


async def _fuse_chunks(queries: List[str], hits: int, k: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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


async def _call_openrouter(context: List[Dict[str, str]], user_message: str, model_id: str) -> str:
    model_id = _resolve_model_id(model_id)
    system_prompt = (
        "You are a helpful assistant. "
        "Answer user's question using only the provided context. "
        "Provide elaborate and informative answers where possible. "
        "If the context is insufficient, say you don't know."
    )
    context_text = "\n\n".join([f"[{c.get('loc','')}] {c.get('chunk','')}" for c in context])
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Grounding context:\n{context_text}\n\nQuestion: {user_message}",
        },
    ]

    try:
        client = _get_llm_client()
        resp = await _create_chat_completion_with_fallback(
            client=client,
            model=model_id,
            messages=messages,
            stream=False,
            enable_reasoning=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return _extract_message_text(resp.choices[0].message.content)


@app.post("/chat")
async def chat(req: ChatRequest):
    """Chat endpoint supporting retrieval, reasoning, and summarization."""
    try:
        return StreamingResponse(_chat_stream(req), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _chat_stream(req: ChatRequest):
    """
    Stream the chat process using Server-Sent Events (SSE).
    Events: 'status', 'thinking', 'queries', 'sources', 'thinking_answer', 'answer'
    """

    def sse(event: str, data: Any) -> str:
        # Ensure data is JSON serializable
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    # 1. Expand queries
    yield sse("status", "Generating search queries...")
    model_id = _resolve_model_id(req.model)
    queries = []

    async for event_type, payload in _prepare_queries_stream(
        req.message, model_id, req.query_k, req.hits, req.k, history=req.history
    ):
        if event_type == "thinking":
            yield sse("thinking", payload)
        elif event_type == "result":
            queries = payload
            yield sse("queries", queries)

    # 2. Retrieve and Fuse
    yield sse("status", "Retrieving context from Vespa...")
    used_queries, chunks = await _fuse_chunks(queries, req.hits, req.k)
    yield sse("sources", chunks)

    if not chunks:
        yield sse("answer", "No relevant context found.")
        yield sse("done", None)
        return

    # 3. Final Generation
    yield sse("status", "Generating answer...")
    system_prompt = (
        "You are a helpful assistant. "
        "Answer the user's question using ONLY the provided context chunks. "
        "If the answer is not in the chunks, say so. "
        "Do not hallucinate. "
    )

    context_text = ""
    for c in chunks:
        context_text += f"Source: {c.get('loc','')}\nContent: {c.get('chunk','')}\n\n"

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    # Add history
    for msg in req.history[-4:]:
        messages.append({"role": msg.get("role"), "content": msg.get("content")})

    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {req.message}",
        }
    )

    client = _get_llm_client()
    try:
        stream = await _create_chat_completion_with_fallback(
            client=client,
            model=model_id,
            messages=messages,
            stream=True,
            enable_reasoning=True,
        )

        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            # Check for reasoning (e.g. DeepSeek R1)
            reasoning = getattr(delta, "reasoning", None)
            reasoning_text = _extract_message_text(reasoning)
            if reasoning_text:
                yield sse("thinking", reasoning_text)

            content = _extract_message_text(getattr(delta, "content", None))
            if content:
                yield sse("answer", content)

        yield sse("done", None)

    except Exception as e:
        yield sse("error", str(e))
