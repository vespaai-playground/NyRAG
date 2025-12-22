import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

from vespa.package import ApplicationPackage

from nyrag.logger import console, logger
from nyrag.utils import _truthy_env


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
    - If `NYRAG_VESPA_ALLOW_CONTENT_CLUSTER_REMOVAL` is truthy: auto-allow.
    - If stdin isn't interactive: auto-deny.
    - Otherwise, ask the user.
    """
    if _truthy_env(os.getenv("NYRAG_VESPA_ALLOW_CONTENT_CLUSTER_REMOVAL", "")):
        logger.warning("NYRAG_VESPA_ALLOW_CONTENT_CLUSTER_REMOVAL=1 set; proceeding with content cluster removal.")
        return True

    if not sys.stdin.isatty():
        logger.warning(
            "Vespa deploy requires 'content-cluster-removal' override, but stdin is not interactive. "
            "Set NYRAG_VESPA_ALLOW_CONTENT_CLUSTER_REMOVAL=1 to proceed."
        )
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
    """

    def _should_fallback(exc: Exception) -> bool:
        msg = str(exc)
        if isinstance(exc, TypeError):
            return True
        return _MISSING_PACKAGE_ERROR in msg

    try:
        return deployer.deploy(application_package=application_package)
    except Exception as e:
        if not _should_fallback(e):
            raise

    try:
        return deployer.deploy(application_root=str(application_root))
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
    return deployer.deploy()


def _set_vespa_endpoint_env_from_app(vespa_app: Any) -> None:
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
    if isinstance(url, str):
        os.environ["VESPA_URL"] = url.rstrip("/")
        if isinstance(port, int):
            os.environ["VESPA_PORT"] = str(port)
        else:
            default_port = "443" if url.strip().lower().startswith("https://") else "8080"
            os.environ.setdefault("VESPA_PORT", default_port)

    # Best-effort: carry over mTLS material from pyvespa if present.
    cert = _as_path_str(getattr(vespa_app, "cert", None))
    key = _as_path_str(getattr(vespa_app, "key", None))
    ca_cert = _as_path_str(getattr(vespa_app, "ca_cert", None))

    if cert and cert.strip():
        os.environ.setdefault("VESPA_CLIENT_CERT", cert.strip())
    if key and key.strip():
        os.environ.setdefault("VESPA_CLIENT_KEY", key.strip())
    if ca_cert and ca_cert.strip():
        os.environ.setdefault("VESPA_CA_CERT", ca_cert.strip())


def deploy_app_package(
    app_dir: Optional[Path],
    *,
    app_package: ApplicationPackage,
) -> bool:
    """
    Deploy the application package using pyvespa deployments.

    Deployment selection:
    - If `NYRAG_LOCAL` is set:
      - truthy => start local Vespa via `VespaDocker`
      - falsy  => deploy to `VespaCloud`
    - If `NYRAG_LOCAL` is not set:
      - if `VESPA_CLOUD_TENANT` or `VESPA_CLOUD_APPLICATION` is set => `VespaCloud`
      - else => `VespaDocker`
    """
    local_env = os.getenv("NYRAG_LOCAL")
    if local_env is not None:
        mode = "docker" if _truthy_env(local_env) else "cloud"
    else:
        has_cloud_hints = bool(
            (os.getenv("VESPA_CLOUD_TENANT") or "").strip() or (os.getenv("VESPA_CLOUD_APPLICATION") or "").strip()
        )
        mode = "cloud" if has_cloud_hints else "docker"

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
                from vespa.deployment import VespaDocker  # type: ignore

                image = os.getenv("NYRAG_VESPA_DOCKER_IMAGE", "vespaengine/vespa:latest")
                logger.info(f"Deploying with VespaDocker (image={image})")

                import inspect

                init_sig = inspect.signature(VespaDocker)
                init_kwargs = {}
                if "image" in init_sig.parameters:
                    init_kwargs["image"] = image
                elif "docker_image" in init_sig.parameters:
                    init_kwargs["docker_image"] = image

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
                return True

            if mode == "cloud":
                from vespa.deployment import VespaCloud  # type: ignore

                tenant = (os.getenv("VESPA_CLOUD_TENANT") or "").strip()
                application_env = (os.getenv("VESPA_CLOUD_APPLICATION") or "").strip()
                application = application_env or app_package.name
                instance = (os.getenv("VESPA_CLOUD_INSTANCE") or "").strip() or "default"
                if not tenant:
                    raise RuntimeError("Missing Vespa Cloud env var: VESPA_CLOUD_TENANT")

                if not application_env:
                    logger.info(f"VESPA_CLOUD_APPLICATION not set; using generated app name '{application}'")
                logger.info(f"Deploying to Vespa Cloud: {tenant}/{application}/{instance}")

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

                api_key_path = (os.getenv("VESPA_CLOUD_API_KEY_PATH") or "").strip() or None
                api_key = (os.getenv("VESPA_CLOUD_API_KEY") or "").strip() or None
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
                _set_vespa_endpoint_env_from_app(vespa_app)
                logger.success("Vespa Cloud deploy succeeded")
                return True

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
                    return False

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
