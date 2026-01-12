"""Tests for the utils module."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from nyrag.config import Config, DeployConfig
from nyrag.defaults import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VESPA_CLOUD_PORT,
    DEFAULT_VESPA_LOCAL_PORT,
    DEFAULT_VESPA_TLS_VERIFY,
)
from nyrag.utils import (
    chunks,
    get_tls_config_from_deploy,
    get_vespa_port,
    is_cloud_mode,
    resolve_vespa_cloud_mtls_paths,
)
from nyrag.vespa_cli import clear_vespa_cli_cache


@pytest.fixture(autouse=True)
def _isolate_cli_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    clear_vespa_cli_cache()
    yield
    clear_vespa_cli_cache()


class TestIsCloudMode:
    """Tests for is_cloud_mode function."""

    def test_cloud_mode(self):
        """Test that cloud mode is detected correctly."""
        deploy_config = DeployConfig(deploy_mode="cloud")
        assert is_cloud_mode(deploy_config) is True

    def test_local_mode(self):
        """Test that local mode is detected correctly."""
        deploy_config = DeployConfig(deploy_mode="local")
        assert is_cloud_mode(deploy_config) is False

    def test_none_deploy_config(self):
        """Test that None deploy config defaults to False."""
        assert is_cloud_mode(None) is False


class TestGetVespaPort:
    """Tests for get_vespa_port function."""

    def test_cloud_mode_default_port(self):
        """Test that cloud mode uses default cloud port."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(
                name="test",
                mode="docs",
                start_loc="/test",
                deploy_mode="cloud",
            )
            port = get_vespa_port(config)
            assert port == DEFAULT_VESPA_CLOUD_PORT

    def test_local_mode_default_port(self):
        """Test that local mode uses default local port."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(
                name="test",
                mode="docs",
                start_loc="/test",
                deploy_mode="local",
            )
            port = get_vespa_port(config)
            assert port == DEFAULT_VESPA_LOCAL_PORT

    def test_explicit_port_override(self):
        """Test that explicit port env var overrides defaults."""
        with patch.dict(os.environ, {"VESPA_PORT": "9090"}):
            config = Config(
                name="test",
                mode="docs",
                start_loc="/test",
                deploy_mode="local",
            )
            port = get_vespa_port(config)
            assert port == 9090

    def test_none_config_default_port(self):
        """Test that None config returns default local port."""
        with patch.dict(os.environ, {}, clear=True):
            port = get_vespa_port(None)
            assert port == DEFAULT_VESPA_LOCAL_PORT


class TestResolveVespaCloudMtlsPaths:
    """Tests for resolve_vespa_cloud_mtls_paths function."""

    def test_mtls_paths(self):
        """Test that mTLS paths are resolved correctly."""
        cert_path, key_path = resolve_vespa_cloud_mtls_paths("my-project")
        expected_base = Path.home() / ".vespa" / "devrel-public.my-project.default"
        assert cert_path == expected_base / "data-plane-public-cert.pem"
        assert key_path == expected_base / "data-plane-private-key.pem"

    def test_mtls_paths_with_target(self):
        """Test that mTLS paths use tenant/app/instance when provided."""
        cert_path, key_path = resolve_vespa_cloud_mtls_paths(
            "ignored-project",
            tenant="tenant",
            application="app",
            instance="instance",
        )
        expected_base = Path.home() / ".vespa" / "tenant.app.instance"
        assert cert_path == expected_base / "data-plane-public-cert.pem"
        assert key_path == expected_base / "data-plane-private-key.pem"


class TestVespaCliFallback:
    """Tests for Vespa CLI fallback behavior."""

    def _write_cli_config(self, data: dict) -> Path:
        config_path = Path.home() / ".vespa" / "cli" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(data))
        clear_vespa_cli_cache()
        return config_path

    def test_cloud_settings_from_cli(self):
        """Test that cloud settings are loaded from Vespa CLI config."""
        self._write_cli_config(
            {
                "current_target": "tenant.app.instance",
                "targets": {
                    "tenant.app.instance": {
                        "type": "cloud",
                        "endpoint": "https://app.tenant.instance.z.vespa-app.cloud:8443",
                        "tenant": "tenant",
                        "application": "app",
                        "instance": "instance",
                        "auth": {
                            "apiKeyPath": "/path/to/api-key.pem",
                            "cert": "/path/to/cert.pem",
                            "key": "/path/to/key.pem",
                            "caCert": "/path/to/ca.pem",
                        },
                    }
                },
            }
        )

        deploy_config = DeployConfig(deploy_mode="cloud")
        assert deploy_config.get_cloud_tenant() == "tenant"
        assert deploy_config.get_cloud_application() == "app"
        assert deploy_config.get_cloud_instance() == "instance"
        assert deploy_config.get_cloud_api_key_path() == "/path/to/api-key.pem"
        assert deploy_config.get_tls_client_cert() == "/path/to/cert.pem"
        assert deploy_config.get_tls_client_key() == "/path/to/key.pem"
        assert deploy_config.get_tls_ca_cert() == "/path/to/ca.pem"

        config = Config(name="test", mode="docs", start_loc="/test", deploy_mode="cloud")
        assert config.get_vespa_url() == "https://app.tenant.instance.z.vespa-app.cloud"
        assert config.get_vespa_port() == 8443

    def test_env_overrides_cli(self):
        """Test that env vars override Vespa CLI config."""
        self._write_cli_config(
            {
                "current_target": "tenant.app.instance",
                "targets": {
                    "tenant.app.instance": {
                        "type": "cloud",
                        "tenant": "tenant",
                        "application": "app",
                        "instance": "instance",
                    }
                },
            }
        )

        with patch.dict(os.environ, {"VESPA_CLOUD_TENANT": "env-tenant"}):
            deploy_config = DeployConfig(deploy_mode="cloud")
            assert deploy_config.get_cloud_tenant() == "env-tenant"

    def test_cloud_settings_from_config(self):
        """Test that config values are used when env vars are missing."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(
                name="test",
                mode="docs",
                start_loc="/test",
                deploy_mode="cloud",
                cloud_tenant="tenant",
            )
            deploy_config = config.get_deploy_config()
            assert deploy_config.get_cloud_tenant() == "tenant"
            assert deploy_config.get_cloud_application() == config.get_app_package_name()
            assert deploy_config.get_cloud_instance() == "default"
            assert (
                config.get_vespa_url() == f"https://{config.get_app_package_name()}.tenant.default.z.vespa-app.cloud"
            )
            assert config.get_vespa_port() == DEFAULT_VESPA_CLOUD_PORT

    def test_team_api_key_env(self):
        """Test that VESPA_TEAM_API_KEY is accepted as an API key."""
        with patch.dict(os.environ, {"VESPA_TEAM_API_KEY": "team-key"}, clear=True):
            deploy_config = DeployConfig(deploy_mode="cloud")
            assert deploy_config.get_cloud_api_key() == "team-key"

    def test_config_endpoint_override(self):
        """Test that config vespa_url/vespa_port override defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(
                name="test",
                mode="docs",
                start_loc="/test",
                deploy_mode="cloud",
                vespa_url="https://example.vespa-app.cloud",
                vespa_port=8443,
            )
            assert config.get_vespa_url() == "https://example.vespa-app.cloud"
            assert config.get_vespa_port() == 8443


class TestGetTlsConfigFromDeploy:
    """Tests for get_tls_config_from_deploy function."""

    def test_no_deploy_config(self):
        """Test with no deploy config (reads from env vars)."""
        with patch.dict(os.environ, {}, clear=True):
            cert, key, ca, verify = get_tls_config_from_deploy(None)
            assert cert is None
            assert key is None
            assert ca is None
            assert verify == DEFAULT_VESPA_TLS_VERIFY

    def test_with_tls_config(self):
        """Test with TLS config from env vars."""
        with patch.dict(
            os.environ,
            {
                "VESPA_CLIENT_CERT": "/path/to/cert.pem",
                "VESPA_CLIENT_KEY": "/path/to/key.pem",
            },
        ):
            deploy_config = DeployConfig(deploy_mode="local")
            cert, key, ca, verify = get_tls_config_from_deploy(deploy_config)
            assert cert == "/path/to/cert.pem"
            assert key == "/path/to/key.pem"
            assert ca is None
            assert verify == DEFAULT_VESPA_TLS_VERIFY

    def test_with_ca_cert(self):
        """Test with CA cert from env var."""
        with patch.dict(os.environ, {"VESPA_CA_CERT": "/path/to/ca.pem"}, clear=True):
            deploy_config = DeployConfig(deploy_mode="local")
            cert, key, ca, verify = get_tls_config_from_deploy(deploy_config)
            assert ca == "/path/to/ca.pem"

    def test_verify_false(self):
        """Test with verify=false from env var."""
        with patch.dict(os.environ, {"VESPA_TLS_VERIFY": "0"}, clear=True):
            deploy_config = DeployConfig(deploy_mode="local")
            cert, key, ca, verify = get_tls_config_from_deploy(deploy_config)
            assert verify is False


class TestChunks:
    """Tests for chunks utility function."""

    def test_basic_chunking(self):
        """Test basic text chunking without overlap."""
        text = " ".join(["word"] * 100)  # 100 words
        result = list(chunks(text, chunk_size=10, overlap=0))
        assert len(result) == 10
        assert all(len(chunk.split()) == 10 for chunk in result)

    def test_chunking_with_overlap(self):
        """Test text chunking with overlap."""
        text = " ".join(["word"] * 50)  # 50 words
        result = list(chunks(text, chunk_size=10, overlap=5))
        assert len(result) > 5

    def test_text_shorter_than_chunk_size(self):
        """Test with text shorter than chunk size."""
        text = "short text"
        result = list(chunks(text, chunk_size=10, overlap=0))
        assert len(result) == 1
        assert result[0] == "short text"

    def test_empty_text(self):
        """Test with empty text."""
        text = ""
        result = list(chunks(text, chunk_size=10, overlap=0))
        assert len(result) == 1
        assert result[0] == ""

    def test_exact_chunk_size(self):
        """Test with text exactly matching chunk size."""
        text = " ".join(["word"] * 10)
        result = list(chunks(text, chunk_size=10, overlap=0))
        assert len(result) == 1

    def test_overlap_larger_than_chunk_size(self):
        """Test with overlap larger than chunk size (edge case)."""
        text = " ".join(["word"] * 100)
        # When overlap >= chunk_size, it should raise ValueError
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            list(chunks(text, chunk_size=10, overlap=15))

    def test_word_boundary_preservation(self):
        """Test that chunks work on word boundaries."""
        text = "hello world this is a test with more words"
        result = list(chunks(text, chunk_size=3, overlap=0))
        # Should split on word boundaries
        assert len(result) >= 2


class TestConstants:
    """Tests for module constants."""

    def test_default_embedding_model(self):
        """Test default embedding model constant."""
        assert DEFAULT_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"

    def test_default_embedding_dim(self):
        """Test default embedding dimension constant."""
        assert DEFAULT_EMBEDDING_DIM == 384

    def test_default_ports(self):
        """Test default port constants."""
        assert DEFAULT_VESPA_LOCAL_PORT == 8080
        assert DEFAULT_VESPA_CLOUD_PORT == 443
