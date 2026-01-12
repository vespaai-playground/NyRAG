import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def set_vespa_target_cloud() -> bool:
    """Set Vespa CLI target to cloud mode.

    Returns:
        True if successful, False otherwise.
    """
    if shutil.which("vespa") is None:
        return False

    try:
        result = subprocess.run(
            ["vespa", "config", "set", "target", "cloud"],
            capture_output=True,
            text=True,
            check=False,
        )
        clear_vespa_cli_cache()
        return result.returncode == 0
    except Exception:
        return False


def vespa_auth_login() -> bool:
    """Run vespa auth login for interactive cloud authentication.

    This opens a browser for OAuth authentication with Vespa Cloud.
    Does not require tenant/application - those are configured later in UI.

    Returns:
        True if authentication succeeded, False otherwise.
    """
    if shutil.which("vespa") is None:
        return False

    try:
        # Run vespa auth login interactively (it will open browser)
        result = subprocess.run(
            ["vespa", "auth", "login"],
            check=False,
        )
        clear_vespa_cli_cache()
        return result.returncode == 0
    except Exception:
        return False


def is_vespa_cloud_authenticated() -> bool:
    """Check if user is authenticated with Vespa Cloud.

    Returns:
        True if valid authentication exists, False otherwise.
    """
    # Check for auth.json file which is created after successful login
    auth_file = Path.home() / ".vespa" / "auth.json"
    if auth_file.exists():
        try:
            with open(auth_file, "r") as f:
                data = json.load(f)
            # Check if there's a valid access token
            if data.get("access_token"):
                return True
        except Exception:
            pass

    # Also check API key as fallback
    config = get_vespa_cli_cloud_config()
    if config.get("api_key") or config.get("api_key_path"):
        return True

    return False


def get_vespa_cloud_secret_token() -> Optional[str]:
    """Get Vespa Cloud secret token from environment or CLI config.

    The token is used for data-plane authentication (feeding/querying).

    Returns:
        The secret token if found, None otherwise.
    """
    # First check environment variable
    token = os.getenv("VESPA_CLOUD_SECRET_TOKEN")
    if token:
        return token

    # Check CLI config for token
    config = load_vespa_cli_config()
    if config:
        # Look for token in various locations
        for key in ("secret_token", "secretToken", "data_plane_token", "token"):
            if key in config and isinstance(config[key], str):
                return config[key]

        # Check in auth section
        auth = config.get("auth", {})
        if isinstance(auth, dict):
            for key in ("secret_token", "secretToken", "token"):
                if key in auth and isinstance(auth[key], str):
                    return auth[key]

    return None


def _candidate_cli_config_paths() -> Tuple[Path, ...]:
    home = Path.home()
    return (
        home / ".vespa" / "cli" / "config.json",
        home / ".vespa" / "config.json",
        home / ".vespa" / "cli" / "targets.json",
    )


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_vespa_cli_config() -> Optional[Dict[str, Any]]:
    """Load Vespa CLI config if present."""
    for path in _candidate_cli_config_paths():
        if path.exists():
            data = _read_json(path)
            if data is not None:
                return data
    return None


def _get_current_target_name(config: Dict[str, Any]) -> Optional[str]:
    for key in (
        "current_target",
        "currentTarget",
        "current",
        "target",
        "active_target",
        "activeTarget",
    ):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _select_inline_target(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key in ("target", "current_target", "currentTarget", "current"):
        value = config.get(key)
        if isinstance(value, dict):
            return value
    cloud = config.get("cloud")
    if isinstance(cloud, dict):
        return cloud
    return None


def _select_target(config: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    targets = config.get("targets")
    if isinstance(targets, dict):
        target_name = _get_current_target_name(config)
        if target_name and target_name in targets and isinstance(targets[target_name], dict):
            return target_name, targets[target_name]
        if len(targets) == 1:
            only_name, only_target = next(iter(targets.items()))
            if isinstance(only_target, dict):
                return only_name, only_target
    return _get_current_target_name(config), _select_inline_target(config)


def _pick_str(mapping: Optional[Dict[str, Any]], *keys: str) -> Optional[str]:
    if not mapping:
        return None
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _parse_application_id(raw: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not raw:
        return None, None, None
    for sep in (".", ":", "/"):
        if sep in raw:
            parts = [p for p in raw.split(sep) if p]
            if len(parts) >= 3:
                return parts[0], parts[1], parts[2]
    return None, None, None


def _classify_api_key(value: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not value:
        return None, None
    if "\n" in value or "BEGIN" in value:
        return None, value
    if "/" in value or value.endswith(".pem") or value.startswith("~"):
        return str(Path(value).expanduser()), None
    path = Path(value)
    if path.exists():
        return str(path), None
    return None, value


def get_vespa_cli_cloud_config() -> Dict[str, Optional[str]]:
    """Return Vespa Cloud settings from Vespa CLI, if available."""
    config = load_vespa_cli_config()
    if not config:
        return {}

    target_name, target = _select_target(config)
    if not target:
        return {}

    auth = target.get("auth") if isinstance(target.get("auth"), dict) else {}
    tls = target.get("tls") if isinstance(target.get("tls"), dict) else {}

    endpoint = _pick_str(
        target,
        "endpoint",
        "url",
        "data_plane_url",
        "dataPlaneUrl",
        "dataPlaneEndpoint",
        "endpointUrl",
    )

    tenant = _pick_str(target, "tenant", "tenantName")
    application = _pick_str(target, "application", "app", "applicationName", "application_id", "applicationId")
    instance = _pick_str(target, "instance", "instanceName")

    app_block = target.get("application")
    if isinstance(app_block, dict):
        tenant = tenant or _pick_str(app_block, "tenant", "tenantName")
        application = application or _pick_str(app_block, "application", "name", "applicationName")
        instance = instance or _pick_str(app_block, "instance", "instanceName")

    if application and (not tenant or not instance):
        parsed_tenant, parsed_app, parsed_instance = _parse_application_id(application)
        tenant = tenant or parsed_tenant
        application = parsed_app or application
        instance = instance or parsed_instance

    if not (tenant and application and instance):
        parsed_tenant, parsed_app, parsed_instance = _parse_application_id(target_name)
        tenant = tenant or parsed_tenant
        application = application or parsed_app
        instance = instance or parsed_instance

    api_key = _pick_str(auth, "apiKey", "api_key")
    api_key_path = _pick_str(auth, "apiKeyPath", "api_key_path", "apiKeyFile", "api_key_file")
    if not api_key and not api_key_path:
        api_key = _pick_str(target, "apiKey", "api_key")
        api_key_path = _pick_str(target, "apiKeyPath", "api_key_path", "apiKeyFile", "api_key_file")

    classified_path, classified_key = _classify_api_key(api_key)
    api_key_path = api_key_path or classified_path
    api_key = classified_key or api_key

    cert = _pick_str(auth, "cert", "certificate", "client_cert", "clientCert", "dataPlaneCert", "data_plane_cert")
    key = _pick_str(auth, "key", "private_key", "client_key", "clientKey", "dataPlaneKey", "data_plane_key")
    if not cert:
        cert = _pick_str(
            target, "cert", "certificate", "client_cert", "clientCert", "dataPlaneCert", "data_plane_cert"
        )
    if not key:
        key = _pick_str(target, "key", "private_key", "client_key", "clientKey", "dataPlaneKey", "data_plane_key")

    ca_cert = _pick_str(tls, "caCert", "ca_cert", "ca", "ca_path", "caPath")
    if not ca_cert:
        ca_cert = _pick_str(target, "caCert", "ca_cert", "ca", "ca_path", "caPath")

    return {
        "tenant": tenant,
        "application": application,
        "instance": instance,
        "api_key_path": api_key_path,
        "api_key": api_key,
        "tls_client_cert": cert,
        "tls_client_key": key,
        "tls_ca_cert": ca_cert,
        "endpoint": endpoint,
    }


def clear_vespa_cli_cache() -> None:
    load_vespa_cli_config.cache_clear()


def ensure_vespa_cli_target(tenant: str, application: str, instance: str) -> bool:
    """Ensure Vespa CLI target is set for the given tenant/app/instance."""
    if not (tenant and application and instance):
        return False
    if shutil.which("vespa") is None:
        return False

    current = get_vespa_cli_cloud_config()
    if (
        current
        and current.get("tenant") == tenant
        and current.get("application") == application
        and current.get("instance") == instance
    ):
        return True

    target = f"{tenant}.{application}.{instance}"
    try:
        result = subprocess.run(
            ["vespa", "target", "set", target],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return False

    if result.returncode != 0:
        return False

    clear_vespa_cli_cache()
    return True
