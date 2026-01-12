import sys
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from vespa.package import ApplicationPackage

from nyrag.defaults import DEFAULT_VESPA_DOCKER_IMAGE
from nyrag.logger import console, logger
from nyrag.vespa_docker import resolve_vespa_docker_class


if TYPE_CHECKING:
    from nyrag.config import DeployConfig


@dataclass
class DeployResult:
    """Result from a Vespa deployment with endpoint information."""

    success: bool
    vespa_url: Optional[str] = None
    vespa_port: Optional[int] = None
    mtls_endpoint: Optional[str] = None
    token_endpoint: Optional[str] = None
    cert_path: Optional[str] = None
    key_path: Optional[str] = None


_CLUSTER_REMOVAL_ALLOWLIST_TOKEN = "content-cluster-removal"
_MISSING_PACKAGE_ERROR = "Either application_package or application_root must be set"


def _validation_overrides_xml(*, until: date) -> str:
    return (
        "<validation-overrides>\n"
        f"  <allow until='{until.isoformat()}'>{_CLUSTER_REMOVAL_ALLOWLIST_TOKEN}</allow>\n"
        "</validation-overrides>\n"
    )


def _looks_like_cluster_removal_error(message: str) -> bool:
    if not message:
        return False
    lowered = message.lower()
    if _CLUSTER_REMOVAL_ALLOWLIST_TOKEN in lowered:
        return True
    return "content cluster" in lowered and "removed" in lowered


def _confirm_cluster_removal(message: str, *, until: date) -> bool:
    """
    Return True if it's OK to deploy with `content-cluster-removal` override.

    Behavior:
    - If stdin isn't interactive: auto-deny.
    - Otherwise, ask the user.
    """
    if not sys.stdin.isatty():
        logger.warning("Vespa deploy requires 'content-cluster-removal' override, but stdin is not interactive.")
        return False

    console.print(
        "\nVespa refused this deploy because it would remove an existing content cluster.\n"
        f"- Override: {_CLUSTER_REMOVAL_ALLOWLIST_TOKEN} (until {until.isoformat()})\n"
        "This will cause loss of all data in that cluster."
    )
    if message.strip():
        console.print(f"\nVespa message:\n{message.strip()}\n")
    answer = console.input("Purge existing cluster data and redeploy? [y/N]: ")
    return answer.strip().lower() in {"y", "yes"}


def _write_validation_overrides(app_dir: Path, *, until: date) -> None:
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "validation-overrides.xml").write_text(
        _validation_overrides_xml(until=until),
        encoding="utf-8",
    )


def _deploy_with_pyvespa(deployer: Any, *, application_package: ApplicationPackage, application_root: Path) -> Any:
    """
    Deploy using pyvespa across minor API differences.

    Different pyvespa versions accept either:
    - deploy(application_package=...)
    - deploy(application_root=...)
    - deploy() after providing application_package/application_root on the deployer

    For VespaCloud, we also pass disk_folder to prevent pyvespa from creating
    a folder with the application name in the current working directory.
    """
    import inspect

    def _should_fallback(exc: Exception) -> bool:
        msg = str(exc)
        if isinstance(exc, TypeError):
            return True
        return _MISSING_PACKAGE_ERROR in msg

    def _get_deploy_kwargs() -> dict:
        """Build kwargs for deploy() based on signature, including disk_folder for VespaCloud."""
        deploy_kwargs: dict = {}
        try:
            deploy_sig = inspect.signature(deployer.deploy)
            # Pass disk_folder to prevent VespaCloud from creating a folder in cwd
            if "disk_folder" in deploy_sig.parameters:
                deploy_kwargs["disk_folder"] = str(application_root)
        except Exception:
            pass
        return deploy_kwargs

    deploy_kwargs = _get_deploy_kwargs()

    try:
        return deployer.deploy(application_package=application_package, **deploy_kwargs)
    except Exception as e:
        if not _should_fallback(e):
            raise

    try:
        return deployer.deploy(application_root=str(application_root), **deploy_kwargs)
    except Exception as e:
        if not _should_fallback(e):
            raise

    # Last resort: set attributes if present and call deploy() with no args.
    for attr, value in (
        ("application_package", application_package),
        ("app_package", application_package),
        ("package", application_package),
        ("application_root", str(application_root)),
    ):
        if hasattr(deployer, attr):
            try:
                setattr(deployer, attr, value)
            except Exception:
                pass
    return deployer.deploy(**deploy_kwargs)


def _set_vespa_endpoint_env_from_app(vespa_app: Any) -> None:
    """Extract endpoint info from a deployed Vespa app and set env vars if missing."""
    import os

    def _as_path_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            return os.fspath(value)
        except Exception:
            pass
        if isinstance(value, str):
            return value
        return None

    url = getattr(vespa_app, "url", None)
    port = getattr(vespa_app, "port", None)
    cert = _as_path_str(getattr(vespa_app, "cert", None))
    key = _as_path_str(getattr(vespa_app, "key", None))

    if isinstance(url, str):
        url = url.rstrip("/")
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            if parsed.scheme and parsed.hostname:
                url = f"{parsed.scheme}://{parsed.hostname}"
                if parsed.port and not os.getenv("VESPA_PORT"):
                    os.environ["VESPA_PORT"] = str(parsed.port)
        except Exception:
            pass

    if url and not os.getenv("VESPA_URL"):
        os.environ["VESPA_URL"] = url
    if port and not os.getenv("VESPA_PORT"):
        os.environ["VESPA_PORT"] = str(port)
    if cert and not os.getenv("VESPA_CLIENT_CERT"):
        os.environ["VESPA_CLIENT_CERT"] = cert
    if key and not os.getenv("VESPA_CLIENT_KEY"):
        os.environ["VESPA_CLIENT_KEY"] = key

    # Log the extracted values for debugging
    if isinstance(url, str):
        logger.debug(f"Vespa endpoint: {url}:{port}")
    if cert:
        logger.debug(f"mTLS cert: {cert}")
    if key:
        logger.debug("mTLS key: (set)")


def deploy_app_package(
    app_dir: Optional[Path],
    *,
    app_package: ApplicationPackage,
    deploy_config: Optional["DeployConfig"] = None,
) -> DeployResult:
    """
    Deploy the application package using pyvespa deployments.

    Deployment mode is determined by deploy_config.deploy_mode:
    - "local" => start local Vespa via `VespaDocker`
    - "cloud" => deploy to `VespaCloud`

    Connection settings come from environment variables:
    - VESPA_URL, VESPA_PORT, VESPA_CONFIGSERVER_URL (for local)
    - VESPA_CLOUD_* env vars (for cloud)

    Returns:
        DeployResult with endpoint information for persistence.
    """
    from nyrag.config import DeployConfig

    if deploy_config is None:
        deploy_config = DeployConfig()

    mode = "docker" if deploy_config.is_local_mode() else "cloud"

    attempted_override = False
    while True:
        try:
            tmp: Optional[tempfile.TemporaryDirectory] = None
            effective_app_dir = Path(app_dir) if app_dir is not None else None
            if effective_app_dir is None:
                tmp = tempfile.TemporaryDirectory()
                effective_app_dir = Path(tmp.name)
                app_package.to_files(str(effective_app_dir))

            if mode == "docker":
                VespaDocker = resolve_vespa_docker_class()

                if VespaDocker.__name__ == "ComposeVespaDocker":
                    logger.info("Deploying with ComposeVespaDocker")
                    cfgsrv_url = deploy_config.get_configserver_url()
                    logger.info(f"Deploying via compose config server at {cfgsrv_url}")
                else:
                    logger.info(f"Deploying with VespaDocker (image={DEFAULT_VESPA_DOCKER_IMAGE})")

                import inspect

                init_sig = inspect.signature(VespaDocker)
                init_kwargs = {}
                if "image" in init_sig.parameters:
                    init_kwargs["image"] = DEFAULT_VESPA_DOCKER_IMAGE
                elif "docker_image" in init_sig.parameters:
                    init_kwargs["docker_image"] = DEFAULT_VESPA_DOCKER_IMAGE

                # Pass config server URL for compose deployer
                if "cfgsrv_url" in init_sig.parameters:
                    init_kwargs["cfgsrv_url"] = deploy_config.get_configserver_url()

                # Some pyvespa versions want the application package/root on the deployer instance.
                if "application_package" in init_sig.parameters:
                    init_kwargs["application_package"] = app_package
                if "application_root" in init_sig.parameters:
                    init_kwargs["application_root"] = str(effective_app_dir)

                docker = VespaDocker(**init_kwargs) if init_kwargs else VespaDocker()

                vespa_app = _deploy_with_pyvespa(
                    docker,
                    application_package=app_package,
                    application_root=effective_app_dir,
                )
                _set_vespa_endpoint_env_from_app(vespa_app)
                logger.success("VespaDocker deploy succeeded")

                # Extract endpoint info for result
                url = getattr(vespa_app, "url", None)
                port = getattr(vespa_app, "port", 8080)
                return DeployResult(
                    success=True,
                    vespa_url=url,
                    vespa_port=port,
                )

            if mode == "cloud":
                from vespa.deployment import VespaCloud  # type: ignore

                tenant = deploy_config.get_cloud_tenant()
                if not tenant:
                    raise RuntimeError(
                        "Missing Vespa Cloud tenant. "
                        "Set cloud_tenant in config, VESPA_CLOUD_TENANT, or select a Vespa CLI target."
                    )

                application = deploy_config.get_cloud_application() or app_package.name
                instance = deploy_config.get_cloud_instance()

                if not deploy_config.get_cloud_application():
                    logger.info(f"VESPA_CLOUD_APPLICATION not set; using generated app name '{application}'")
                logger.info(f"Deploying to Vespa Cloud: {tenant}/{application}/{instance}")

                from nyrag.vespa_cli import ensure_vespa_cli_target

                if deploy_config.cloud_tenant:
                    if ensure_vespa_cli_target(tenant, application, instance):
                        logger.info(f"Vespa CLI target set to {tenant}.{application}.{instance}")
                    else:
                        logger.debug("Skipping Vespa CLI target set (CLI not found or command failed)")

                import inspect

                init_sig = inspect.signature(VespaCloud)
                init_kwargs = {}
                for key, value in (
                    ("tenant", tenant),
                    ("application", application),
                    ("instance", instance),
                ):
                    if key in init_sig.parameters:
                        init_kwargs[key] = value

                # Some pyvespa versions require application_root/application_package on the deployer.
                if "application_package" in init_sig.parameters:
                    init_kwargs["application_package"] = app_package
                if "application_root" in init_sig.parameters:
                    init_kwargs["application_root"] = str(effective_app_dir)

                api_key_path = deploy_config.get_cloud_api_key_path()
                api_key = deploy_config.get_cloud_api_key()
                if api_key_path and "api_key_path" in init_sig.parameters:
                    init_kwargs["api_key_path"] = api_key_path
                if api_key and "api_key" in init_sig.parameters:
                    init_kwargs["api_key"] = api_key

                cloud = VespaCloud(**init_kwargs)

                vespa_app = _deploy_with_pyvespa(
                    cloud,
                    application_package=app_package,
                    application_root=effective_app_dir,
                )

                # Get endpoints directly from VespaCloud object (more reliable than parsing logs)
                mtls_endpoint = None
                token_endpoint = None
                try:
                    if hasattr(cloud, "get_mtls_endpoint"):
                        mtls_endpoint = cloud.get_mtls_endpoint()
                        if mtls_endpoint:
                            logger.info(f"mTLS endpoint: {mtls_endpoint}")
                except Exception as e:
                    logger.debug(f"Could not get mTLS endpoint: {e}")

                try:
                    if hasattr(cloud, "get_token_endpoint"):
                        token_endpoint = cloud.get_token_endpoint(instance=instance)
                        if token_endpoint:
                            logger.info(f"Token endpoint: {token_endpoint}")
                except Exception as e:
                    logger.debug(f"Could not get token endpoint: {e}")

                # Determine the best endpoint URL
                endpoint_url = mtls_endpoint or token_endpoint

                # Set environment variables from endpoints
                import os

                if endpoint_url and not os.getenv("VESPA_URL"):
                    os.environ["VESPA_URL"] = endpoint_url.rstrip("/")
                    os.environ["VESPA_PORT"] = "443"

                # Also extract from vespa_app as fallback
                _set_vespa_endpoint_env_from_app(vespa_app)

                # Get cert/key paths from vespa_app
                cert_path = None
                key_path = None
                if hasattr(vespa_app, "cert"):
                    cert_path = getattr(vespa_app, "cert", None)
                if hasattr(vespa_app, "key"):
                    key_path = getattr(vespa_app, "key", None)

                logger.success("Vespa Cloud deploy succeeded")
                return DeployResult(
                    success=True,
                    vespa_url=endpoint_url,
                    vespa_port=443,
                    mtls_endpoint=mtls_endpoint,
                    token_endpoint=token_endpoint,
                    cert_path=cert_path,
                    key_path=key_path,
                )

            raise ValueError(f"Unknown Vespa deploy mode: {mode!r}")
        except Exception as e:
            message = str(e)
            if _looks_like_cluster_removal_error(message) and not attempted_override:
                until = date.today() + timedelta(days=7)
                if not _confirm_cluster_removal(message, until=until):
                    logger.warning(
                        "Skipping Vespa deploy to avoid content cluster removal; "
                        "feeding/query may fail until the app is deployed."
                    )
                    return DeployResult(success=False)

                # Write overrides into the on-disk app package and retry.
                target_dir = Path(app_dir) if app_dir is not None else None
                if target_dir is None:
                    tmp = tempfile.TemporaryDirectory()
                    target_dir = Path(tmp.name)
                    app_package.to_files(str(target_dir))
                _write_validation_overrides(target_dir, until=until)
                app_dir = target_dir

                attempted_override = True
                continue

            raise
