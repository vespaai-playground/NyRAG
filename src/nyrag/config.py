import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field, field_validator

from nyrag.defaults import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DEPLOY_MODE,
    DEFAULT_DISTANCE_METRIC,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_VESPA_CLOUD_INSTANCE,
    DEFAULT_VESPA_CLOUD_PORT,
    DEFAULT_VESPA_CONFIGSERVER_URL,
    DEFAULT_VESPA_LOCAL_PORT,
    DEFAULT_VESPA_TLS_VERIFY,
    DEFAULT_VESPA_URL,
)
from nyrag.vespa_cli import get_vespa_cli_cloud_config


class CrawlParams(BaseModel):
    """Parameters specific to web crawling."""

    respect_robots_txt: bool = True
    aggressive_crawl: bool = False
    follow_subdomains: bool = True
    strict_mode: bool = False
    user_agent_type: Literal["chrome", "firefox", "safari", "mobile", "bot"] = "chrome"
    custom_user_agent: Optional[str] = None
    allowed_domains: Optional[List[str]] = None


class DocParams(BaseModel):
    """Parameters specific to document processing."""

    recursive: bool = True
    include_hidden: bool = False
    follow_symlinks: bool = False
    max_file_size_mb: Optional[float] = None
    file_extensions: Optional[List[str]] = None


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""

    base_url: str = DEFAULT_LLM_BASE_URL
    model: str = DEFAULT_LLM_MODEL
    api_key: Optional[str] = None


class DeployConfig(BaseModel):
    """Configuration for deployment settings.

    deploy_mode is set in the config file at top level.
    All connection settings come from environment variables with defaults:

    Local mode env vars:
        VESPA_URL: Vespa endpoint URL (default: http://localhost)
        VESPA_PORT: Vespa port (default: 8080)
        VESPA_CONFIGSERVER_URL: Config server URL for compose deploy

    Cloud mode env vars:
        VESPA_CLOUD_TENANT: Vespa Cloud tenant (required)
        VESPA_CLOUD_APPLICATION: Application name (optional, auto-generated)
        VESPA_CLOUD_INSTANCE: Instance name (default: default)
        VESPA_CLOUD_API_KEY_PATH: Path to API key file
        VESPA_CLOUD_API_KEY: API key content (alternative to path)
        VESPA_TEAM_API_KEY: Team API key (preferred)
        VESPA_CLIENT_CERT: mTLS client certificate path
        VESPA_CLIENT_KEY: mTLS client key path
        VESPA_CA_CERT: CA certificate path
        VESPA_TLS_VERIFY: TLS verification (default: 1)

    Vespa CLI fallback:
        If env vars are not set, values are read from Vespa CLI config
        (e.g. after `vespa auth login` and selecting a target).
    """

    deploy_mode: Literal["local", "cloud"] = DEFAULT_DEPLOY_MODE
    cloud_tenant: Optional[str] = None
    cloud_application: Optional[str] = None
    cloud_instance: Optional[str] = None

    def get_vespa_url(self) -> str:
        """Get Vespa URL from env var or default."""
        env_url = os.getenv("VESPA_URL")
        if env_url:
            return env_url
        if self.deploy_mode == "cloud":
            if self.cloud_tenant:
                tenant = self.get_cloud_tenant()
                application = self.get_cloud_application()
                instance = self.get_cloud_instance()
                if tenant and application and instance:
                    return f"https://{application}.{tenant}.{instance}.z.vespa-app.cloud"
            cli = get_vespa_cli_cloud_config()
            endpoint = cli.get("endpoint")
            if endpoint:
                parsed = urlparse(endpoint)
                if parsed.scheme and parsed.hostname:
                    return f"{parsed.scheme}://{parsed.hostname}"
                return endpoint.rstrip("/")
            tenant = self.get_cloud_tenant()
            application = self.get_cloud_application()
            instance = self.get_cloud_instance()
            if tenant and application and instance:
                return f"https://{application}.{tenant}.{instance}.z.vespa-app.cloud"
        return DEFAULT_VESPA_URL

    def get_vespa_port(self) -> int:
        """Get Vespa port from env var or default based on mode."""
        port_str = os.getenv("VESPA_PORT")
        if port_str:
            return int(port_str)
        if self.deploy_mode == "cloud":
            endpoint = None
            if not self.cloud_tenant:
                cli = get_vespa_cli_cloud_config()
                endpoint = cli.get("endpoint")
            if endpoint:
                parsed = urlparse(endpoint)
                if parsed.port:
                    return parsed.port
        return DEFAULT_VESPA_CLOUD_PORT if self.deploy_mode == "cloud" else DEFAULT_VESPA_LOCAL_PORT

    def get_configserver_url(self) -> str:
        """Get Vespa config server URL from env var or default."""
        return os.getenv("VESPA_CONFIGSERVER_URL", DEFAULT_VESPA_CONFIGSERVER_URL)

    def get_cloud_tenant(self) -> Optional[str]:
        """Get Vespa Cloud tenant from env var."""
        env_value = os.getenv("VESPA_CLOUD_TENANT")
        if env_value:
            return env_value
        if self.cloud_tenant:
            return self.cloud_tenant
        return get_vespa_cli_cloud_config().get("tenant")

    def get_cloud_application(self) -> Optional[str]:
        """Get Vespa Cloud application from env var."""
        env_value = os.getenv("VESPA_CLOUD_APPLICATION")
        if env_value:
            return env_value
        if self.cloud_application:
            return self.cloud_application
        return get_vespa_cli_cloud_config().get("application")

    def get_cloud_instance(self) -> str:
        """Get Vespa Cloud instance from env var or default."""
        env_value = os.getenv("VESPA_CLOUD_INSTANCE")
        if env_value:
            return env_value
        if self.cloud_instance:
            return self.cloud_instance
        cli_value = get_vespa_cli_cloud_config().get("instance")
        return cli_value or DEFAULT_VESPA_CLOUD_INSTANCE

    def get_cloud_api_key_path(self) -> Optional[str]:
        """Get Vespa Cloud API key path from env var."""
        env_value = os.getenv("VESPA_CLOUD_API_KEY_PATH")
        if env_value:
            return env_value
        return get_vespa_cli_cloud_config().get("api_key_path")

    def get_cloud_api_key(self) -> Optional[str]:
        """Get Vespa Cloud API key content from env var."""
        env_value = os.getenv("VESPA_CLOUD_API_KEY")
        if env_value:
            return env_value
        team_key = os.getenv("VESPA_TEAM_API_KEY")
        if team_key:
            return team_key
        return get_vespa_cli_cloud_config().get("api_key")

    def get_tls_client_cert(self) -> Optional[str]:
        """Get mTLS client certificate path from env var."""
        env_value = os.getenv("VESPA_CLIENT_CERT")
        if env_value:
            return env_value
        if self.deploy_mode != "cloud":
            return None
        return get_vespa_cli_cloud_config().get("tls_client_cert")

    def get_tls_client_key(self) -> Optional[str]:
        """Get mTLS client key path from env var."""
        env_value = os.getenv("VESPA_CLIENT_KEY")
        if env_value:
            return env_value
        if self.deploy_mode != "cloud":
            return None
        return get_vespa_cli_cloud_config().get("tls_client_key")

    def get_tls_ca_cert(self) -> Optional[str]:
        """Get CA certificate path from env var."""
        env_value = os.getenv("VESPA_CA_CERT")
        if env_value:
            return env_value
        if self.deploy_mode != "cloud":
            return None
        return get_vespa_cli_cloud_config().get("tls_ca_cert")

    def get_tls_verify(self) -> bool:
        """Get TLS verification setting from env var or default."""
        verify_str = os.getenv("VESPA_TLS_VERIFY")
        if verify_str is not None:
            return verify_str.strip().lower() in ("1", "true", "yes")
        return DEFAULT_VESPA_TLS_VERIFY

    def get_cloud_secret_token(self) -> Optional[str]:
        """Get Vespa Cloud secret token for data-plane authentication.

        Priority:
        1. Environment variable VESPA_CLOUD_SECRET_TOKEN
        2. Vespa CLI config

        Returns:
            The secret token if found, None otherwise.
        """
        env_value = os.getenv("VESPA_CLOUD_SECRET_TOKEN")
        if env_value:
            return env_value
        # Vespa CLI config is checked in vespa_cli module
        return None

    def is_cloud_mode(self) -> bool:
        """Check if deployment mode is cloud."""
        return self.deploy_mode == "cloud"

    def is_local_mode(self) -> bool:
        """Check if deployment mode is local (Docker)."""
        return self.deploy_mode == "local"


class RAGParams(BaseModel):
    """Configuration for RAG parameters with defaults."""

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    distance_metric: str = DEFAULT_DISTANCE_METRIC
    max_tokens: Optional[int] = None


class Config(BaseModel):
    """Configuration model for nyrag.

    deploy_mode controls deployment target:
    - "local": Deploy to local Docker (default)
    - "cloud": Deploy to Vespa Cloud

    Connection settings come from environment variables.
    For cloud mode, you can set cloud_tenant
    in config to avoid env vars (env still wins when set).
    You can also set vespa_url/vespa_port in config; env vars still take precedence.

    Hidden fields (vespa_host, vespa_port_resolved, vespa_cloud_token_id) are
    persisted in conf.yml for inference but excluded from the UI editor schema.
    """

    name: str
    mode: Literal["web", "docs"]
    start_loc: str
    deploy_mode: Literal["local", "cloud"] = DEFAULT_DEPLOY_MODE
    cloud_tenant: Optional[str] = None
    vespa_url: Optional[str] = None
    vespa_port: Optional[int] = None
    exclude: Optional[List[str]] = None
    rag_params: Optional[RAGParams] = None
    crawl_params: Optional[CrawlParams] = None
    doc_params: Optional[DocParams] = None
    llm_config: Optional[LLMConfig] = None

    # Hidden vespa connection fields - persisted for inference, not shown in UI editor
    # These are populated after deployment and used for feeding/querying
    vespa_host: Optional[str] = Field(default=None, exclude=True)
    vespa_port_resolved: Optional[int] = Field(default=None, exclude=True)
    vespa_cloud_token_id: Optional[str] = Field(default=None, exclude=True)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate and normalize mode."""
        if v.lower() in ["web", "docs", "doc"]:
            return "docs" if v.lower() in ["docs", "doc"] else "web"
        raise ValueError("mode must be 'web' or 'docs'")

    def model_post_init(self, __context):
        """Initialize params with defaults if None."""
        if self.crawl_params is None:
            self.crawl_params = CrawlParams()
        if self.doc_params is None:
            self.doc_params = DocParams()
        if self.rag_params is None:
            self.rag_params = RAGParams()
        if self.llm_config is None:
            self.llm_config = LLMConfig()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def get_output_path(self) -> Path:
        """Get the output directory path."""
        # Use schema name format for consistency (lowercase alphanumeric only)
        schema_name = self.get_schema_name()
        return Path("output") / schema_name

    def get_app_path(self) -> Path:
        """Get the app directory path for Vespa schema."""
        return self.get_output_path() / "app"

    def get_schema_name(self) -> str:
        """Get the schema name in format nyragPROJECTNAME (lowercase alphanumeric only)."""
        # Remove hyphens, underscores, and convert to lowercase for valid Vespa schema name
        clean_name = self.name.replace("-", "").replace("_", "").lower()
        return f"nyrag{clean_name}"

    def get_app_package_name(self) -> str:
        """Get a valid application package name (lowercase, no hyphens, max 20 chars)."""
        # Remove hyphens and convert to lowercase
        clean_name = self.name.replace("-", "").replace("_", "").lower()
        # Prefix with nyrag and limit to 20 characters
        app_name = f"nyrag{clean_name}"[:20]
        return app_name

    def get_schema_params(self) -> Dict[str, Any]:
        """Get schema parameters from rag_params."""
        if self.rag_params is None:
            return {
                "embedding_dim": DEFAULT_EMBEDDING_DIM,
                "chunk_size": DEFAULT_CHUNK_SIZE,
                "distance_metric": DEFAULT_DISTANCE_METRIC,
            }
        return {
            "embedding_dim": self.rag_params.embedding_dim,
            "chunk_size": self.rag_params.chunk_size,
            "distance_metric": self.rag_params.distance_metric,
        }

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration from llm_config section."""
        if self.llm_config:
            return {
                "llm_base_url": self.llm_config.base_url,
                "llm_model": self.llm_config.model,
                "llm_api_key": self.llm_config.api_key,
            }

        return {
            "llm_base_url": DEFAULT_LLM_BASE_URL,
            "llm_model": DEFAULT_LLM_MODEL,
            "llm_api_key": None,
        }

    def get_deploy_config(self) -> DeployConfig:
        """Get deployment configuration."""
        cloud_application = None
        cloud_instance = None
        if self.deploy_mode == "cloud" and self.cloud_tenant:
            cloud_application = self.get_app_package_name()
            cloud_instance = DEFAULT_VESPA_CLOUD_INSTANCE
        return DeployConfig(
            deploy_mode=self.deploy_mode,
            cloud_tenant=self.cloud_tenant,
            cloud_application=cloud_application,
            cloud_instance=cloud_instance,
        )

    def get_vespa_url(self) -> str:
        """Get Vespa URL from deploy config (reads from env var)."""
        env_url = os.getenv("VESPA_URL")
        if env_url:
            return env_url
        if self.vespa_url:
            return self.vespa_url
        return self.get_deploy_config().get_vespa_url()

    def get_vespa_port(self) -> int:
        """Get Vespa port from deploy config (reads from env var)."""
        port_str = os.getenv("VESPA_PORT")
        if port_str:
            return int(port_str)
        if self.vespa_port is not None:
            return int(self.vespa_port)
        return self.get_deploy_config().get_vespa_port()

    def is_cloud_mode(self) -> bool:
        """Check if deployment mode is cloud."""
        return self.deploy_mode == "cloud"

    def is_local_deploy_mode(self) -> bool:
        """Check if deployment mode is local (Docker)."""
        return self.deploy_mode == "local"

    def get_embedding_model(self) -> str:
        """Get embedding model from rag_params."""
        if self.rag_params is None:
            return DEFAULT_EMBEDDING_MODEL
        return self.rag_params.embedding_model

    def is_web_mode(self) -> bool:
        """Check if config is for web crawling."""
        return self.mode == "web"

    def is_docs_mode(self) -> bool:
        """Check if config is for document processing."""
        return self.mode == "docs"


def get_config_options(mode: str = "web") -> Dict[str, Any]:
    """
    Return the interactive configuration schema for the frontend.
    Dynamically hides irrelevant sections (crawl_params for docs, doc_params for web).
    """
    # Common base fields
    schema = {
        "name": {"type": "string", "label": "name"},
        "mode": {"type": "select", "label": "mode", "options": ["web", "docs"]},
        "start_loc": {"type": "string", "label": "start_loc"},
        "deploy_mode": {
            "type": "select",
            "label": "deploy_mode",
            "options": ["local", "cloud"],
        },
        "cloud_tenant": {"type": "string", "label": "cloud_tenant", "optional": True},
        "exclude": {"type": "list", "label": "exclude"},
    }

    # Web Mode Specifics
    if mode == "web":
        schema["crawl_params"] = {
            "type": "nested",
            "label": "crawl_params",
            "fields": {
                "respect_robots_txt": {
                    "type": "boolean",
                    "label": "respect_robots_txt",
                },
                "follow_subdomains": {"type": "boolean", "label": "follow_subdomains"},
                "user_agent_type": {
                    "type": "select",
                    "label": "user_agent_type",
                    "options": ["chrome", "firefox", "bot"],
                },
                "aggressive": {"type": "boolean", "label": "aggressive"},
                "strict_mode": {"type": "boolean", "label": "strict_mode"},
                "custom_user_agent": {"type": "string", "label": "custom_user_agent"},
                "allowed_domains": {"type": "list", "label": "allowed_domains"},
            },
        }

    # Doc Mode Specifics
    if mode == "docs":
        schema["doc_params"] = {
            "type": "nested",
            "label": "doc_params",
            "fields": {
                "recursive": {"type": "boolean", "label": "recursive"},
                "include_hidden": {"type": "boolean", "label": "include_hidden"},
                "follow_symlinks": {"type": "boolean", "label": "follow_symlinks"},
                "max_file_size_mb": {"type": "number", "label": "max_file_size_mb"},
                "file_extensions": {"type": "list", "label": "file_extensions"},
            },
        }

    # Always include RAG and LLM params
    schema["rag_params"] = {
        "type": "nested",
        "label": "rag_params",
        "fields": {
            "embedding_model": {"type": "string", "label": "embedding_model"},
            "embedding_dim": {"type": "number", "label": "embedding_dim"},
            "chunk_size": {"type": "number", "label": "chunk_size"},
            "chunk_overlap": {"type": "number", "label": "chunk_overlap"},
            "distance_metric": {
                "type": "select",
                "label": "distance_metric",
                "options": ["angular", "euclidean", "dot", "hamming"],
            },
        },
    }

    schema["llm_config"] = {
        "type": "nested",
        "label": "llm_config",
        "optional": True,
        "fields": {
            "base_url": {"type": "string", "label": "base_url"},
            "model": {"type": "string", "label": "model"},
            "api_key": {"type": "string", "label": "api_key", "masked": True},
        },
    }

    return schema


def get_example_configs() -> Dict[str, str]:
    """
    Return available template configurations from the package.
    Returns a dict of {name: yaml_content}.
    """
    import importlib.resources as pkg_resources

    examples: Dict[str, str] = {}
    try:
        # Python 3.9+ style
        files = pkg_resources.files("nyrag.examples")
        for item in files.iterdir():
            if item.name.endswith(".yml") or item.name.endswith(".yaml"):
                name = item.name.rsplit(".", 1)[0]
                examples[name] = item.read_text()
    except Exception:
        # Fallback for older Python or missing package
        pass

    return examples
