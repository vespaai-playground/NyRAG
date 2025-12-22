import inspect
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_LOCAL_PORT = 8080
DEFAULT_CLOUD_PORT = 443
DEFAULT_CLOUD_CERT_NAME = "data-plane-public-cert.pem"
DEFAULT_CLOUD_KEY_NAME = "data-plane-private-key.pem"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIM = 384


def _truthy_env(value: str) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def is_vespa_cloud(vespa_url: str) -> bool:
    local_env = os.getenv("NYRAG_LOCAL")
    if local_env is not None:
        return not _truthy_env(local_env)
    return (vespa_url or "").strip().lower().startswith("https://")


def resolve_vespa_port(vespa_url: str) -> int:
    port_env = (os.getenv("VESPA_PORT") or "").strip()
    if port_env:
        return int(port_env)
    return DEFAULT_CLOUD_PORT if is_vespa_cloud(vespa_url) else DEFAULT_LOCAL_PORT


def resolve_vespa_cloud_mtls_paths(project_folder: str) -> Tuple[Path, Path]:
    base_dir = Path.home() / ".vespa" / f"devrel-public.{project_folder}.default"
    return base_dir / DEFAULT_CLOUD_CERT_NAME, base_dir / DEFAULT_CLOUD_KEY_NAME


def get_vespa_tls_config() -> Tuple[Optional[str], Optional[str], Optional[str], Optional[object]]:
    """Get Vespa TLS configuration from environment variables.

    Returns:
        Tuple of (cert_path, key_path, ca_cert, verify)
    """
    cert_path = (os.getenv("VESPA_CLIENT_CERT") or "").strip() or None
    key_path = (os.getenv("VESPA_CLIENT_KEY") or "").strip() or None
    ca_cert = (os.getenv("VESPA_CA_CERT") or "").strip() or None

    verify_env = (os.getenv("VESPA_TLS_VERIFY") or "").strip().lower()
    verify: Optional[object]
    if verify_env in {"0", "false", "no", "off"}:
        verify = False
    elif verify_env:
        verify = verify_env
    else:
        verify = None

    return cert_path, key_path, ca_cert, verify


def make_vespa_client(
    vespa_url: str,
    vespa_port: int,
    cert_path: Optional[str] = None,
    key_path: Optional[str] = None,
    ca_cert: Optional[str] = None,
    verify: Optional[object] = None,
) -> Any:
    """Create a Vespa client with proper configuration for different pyvespa versions.

    Args:
        vespa_url: The Vespa endpoint URL
        vespa_port: The Vespa port
        cert_path: Path to client certificate (optional)
        key_path: Path to client key (optional)
        ca_cert: Path to CA certificate (optional)
        verify: TLS verification setting (optional)

    Returns:
        Configured Vespa client instance
    """
    from vespa.application import Vespa

    kwargs: Dict[str, Any] = {}
    try:
        sig = inspect.signature(Vespa)
    except Exception:
        sig = None

    endpoint = f"{vespa_url}:{vespa_port}"
    if sig and "endpoint" in sig.parameters:
        kwargs["endpoint"] = endpoint
    else:
        kwargs["url"] = vespa_url
        kwargs["port"] = vespa_port

    if cert_path and key_path and sig and "cert" in sig.parameters:
        if "key" in sig.parameters:
            kwargs["cert"] = cert_path
            kwargs["key"] = key_path
        else:
            kwargs["cert"] = (cert_path, key_path)

    if ca_cert and sig and "ca_cert" in sig.parameters:
        kwargs["ca_cert"] = ca_cert
    if verify is not None and sig and "verify" in sig.parameters:
        kwargs["verify"] = verify

    return Vespa(**kwargs)


def chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into chunks of specified size with optional overlap.

    Args:
        text: The input text to split
        chunk_size: Size of each chunk (in words)
        overlap: Number of overlapping words between chunks

    Returns:
        List of text chunks
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    words = text.split()
    word_count = len(words)

    if word_count <= chunk_size:
        return [text]

    chunk_list = []
    start = 0
    while start < word_count:
        end = min(start + chunk_size, word_count)
        chunk_list.append(" ".join(words[start:end]))
        start += chunk_size - overlap

    return chunk_list
