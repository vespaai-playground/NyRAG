# NyRAG

NyRAG (pronounced as knee-RAG) is a simple tool for building RAG applications by crawling websites or processing documents, then deploying to Vespa for hybrid search with an integrated chat UI.

![NyRAG Chat UI](assets/ui.png)

## How It Works

When a user asks a question, NyRAG performs a multi-stage retrieval process:

1. **Query Enhancement**: An LLM generates additional search queries based on the user's question and initial context to improve retrieval coverage
2. **Embedding Generation**: Each query is converted to embeddings using the configured SentenceTransformer model
3. **Vespa Search**: Queries are executed against Vespa using nearestNeighbor search with the `best_chunk_score` ranking profile to find the most relevant document chunks
4. **Chunk Fusion**: Results from all queries are aggregated, deduplicated, and ranked by score to select the top-k most relevant chunks
5. **Answer Generation**: The retrieved context is sent to an LLM which generates a grounded answer based only on the provided chunks

This multi-query RAG approach with chunk-level retrieval ensures answers are comprehensive and grounded in your actual content, whether from crawled websites or processed documents.

### LLM Support

NyRAG works with **any OpenAI-compatible API**, including:
- **OpenRouter** (100+ models from various providers)
- **Ollama** (local models: Llama, Mistral, Qwen, etc.)
- **LM Studio** (local GUI for running models)
- **vLLM** (high-performance local or remote inference)
- **LocalAI** (local OpenAI drop-in replacement)
- **OpenAI** (GPT-4, GPT-3.5, etc.)
- Any other service implementing the OpenAI API format


## Installation

```bash
pip install nyrag
```

We recommend `uv`:

```bash
uv init --python 3.10
uv venv
uv sync
source .venv/bin/activate
uv pip install -U nyrag
```

For development:

```bash
git clone https://github.com/abhishekkrthakur/nyrag.git
cd nyrag
pip install -e .
```

## Usage

NyRAG is designed to be used primarily through its web UI, which manages the entire lifecycle from data processing to chat.

### 1. Start the UI

**Local Mode** (requires Docker):
```bash
nyrag ui
```

**Cloud Mode** (requires Vespa Cloud account):
```bash
nyrag ui --cloud
```

Open http://localhost:8000 in your browser.

### 2. Configure & Process

In the UI, you can create a new configuration for your data source.

**Example Web Crawl Config:**

```yaml
name: mywebsite
mode: web
start_loc: https://example.com/
crawl_params:
  respect_robots_txt: true
rag_params:
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
```

**Example Docs Processing Config:**

```yaml
name: mydocs
mode: docs
start_loc: /path/to/documents/
doc_params:
  recursive: true
rag_params:
  embedding_model: sentence-transformers/all-mpnet-base-v2
```

### 3. Chat

Once processing is complete, you can start chatting with your data immediately in the UI. Make sure your configuration includes your LLM API key and model selection.


## Configuration Reference

### Cloud Deploy Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cloud_tenant` | str | `None` | Vespa Cloud tenant (required for cloud mode if no env/CLI target) |

### Connection Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vespa_url` | str | `None` | Vespa endpoint URL (auto-filled into `conf.yml` after deploy) |
| `vespa_port` | int | `None` | Vespa endpoint port (auto-filled into `conf.yml` after deploy) |

### Web Mode Parameters (`crawl_params`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `respect_robots_txt` | bool | `true` | Respect robots.txt rules |
| `aggressive_crawl` | bool | `false` | Faster crawling with more concurrent requests |
| `follow_subdomains` | bool | `true` | Follow links to subdomains |
| `strict_mode` | bool | `false` | Only crawl URLs matching start pattern |
| `user_agent_type` | str | `chrome` | `chrome`, `firefox`, `safari`, `mobile`, `bot` |
| `custom_user_agent` | str | `None` | Custom user agent string |
| `allowed_domains` | list | `None` | Explicitly allowed domains |

### Docs Mode Parameters (`doc_params`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `recursive` | bool | `true` | Process subdirectories |
| `include_hidden` | bool | `false` | Include hidden files |
| `follow_symlinks` | bool | `false` | Follow symbolic links |
| `max_file_size_mb` | float | `None` | Max file size in MB |
| `file_extensions` | list | `None` | Only process these extensions |

### RAG Parameters (`rag_params`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_model` | str | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `embedding_dim` | int | `384` | Embedding dimension |
| `chunk_size` | int | `1024` | Chunk size for text splitting |
| `chunk_overlap` | int | `50` | Overlap between chunks |
| `distance_metric` | str | `angular` | Distance metric |
| `max_tokens` | int | `8192` | Max tokens per document |
| `llm_base_url` | str | `None` | LLM API base URL (OpenAI-compatible) |
| `llm_model` | str | `None` | LLM model name |
| `llm_api_key` | str | `None` | LLM API key |

---


## LLM Provider Support

NyRAG works with any OpenAI-compatible API. Just configure the `rag_params` in your UI settings.

| Provider | Base URL | Model Example | API Key |
|----------|----------|---------------|---------|
| **Ollama** | `http://localhost:11434/v1` | `llama3.2` | `dummy` |
| **LM Studio** | `http://localhost:1234/v1` | `local-model` | `dummy` |
| **vLLM** | `http://localhost:8000/v1` | `meta-llama/Llama-3.2-3B-Instruct` | `dummy` |
| **OpenRouter** | `https://openrouter.ai/api/v1` | `openai/gpt-5.2` | `your-key` |
| **OpenAI** | `None` (default) | `openai/gpt-4o` | `your-key` |

**Example Config:**

```yaml
llm_config:
  llm_base_url: https://openrouter.ai/api/v1
  llm_model: llama3.2
  llm_api_key: dummy
```

---
