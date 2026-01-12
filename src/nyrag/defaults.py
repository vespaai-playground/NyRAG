"""Default values for optional configuration keys.

Config file only contains deploy_mode. Connection settings come from
environment variables with these defaults.
"""

from typing import Literal


# Deploy mode: "local" (Docker) or "cloud" (Vespa Cloud)
# This is a static default. The actual runtime default might be "cloud"
# if running with --cloud flag (NYRAG_CLOUD_MODE=1)
DEFAULT_DEPLOY_MODE: Literal["local", "cloud"] = "local"

# Vespa connection defaults
DEFAULT_VESPA_URL = "http://localhost"
DEFAULT_VESPA_LOCAL_PORT = 8080
DEFAULT_VESPA_CLOUD_PORT = 443
DEFAULT_VESPA_CONFIGSERVER_URL = "http://localhost:19071"

# Vespa Docker defaults
DEFAULT_VESPA_DOCKER_IMAGE = "vespaengine/vespa:latest"

# Vespa Cloud defaults
DEFAULT_VESPA_CLOUD_INSTANCE = "default"

# TLS defaults
DEFAULT_VESPA_TLS_VERIFY = True

# Vespa Cloud mTLS file names
DEFAULT_CLOUD_CERT_NAME = "data-plane-public-cert.pem"
DEFAULT_CLOUD_KEY_NAME = "data-plane-private-key.pem"

# RAG defaults
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_DISTANCE_METRIC = "angular"

# LLM defaults
DEFAULT_LLM_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_MODEL = "anthropic/claude-3.5-sonnet"
