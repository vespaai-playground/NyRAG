import uuid
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer
from vespa.io import VespaResponse

from nyrag.config import Config
from nyrag.defaults import DEFAULT_EMBEDDING_MODEL, DEFAULT_VESPA_LOCAL_PORT, DEFAULT_VESPA_URL
from nyrag.deploy import deploy_app_package
from nyrag.logger import logger
from nyrag.schema import VespaSchema
from nyrag.utils import chunks, get_cloud_secret_token, get_tls_config_from_deploy, make_vespa_client


class VespaFeeder:
    """Feed processed data (content, chunks, embeddings) into Vespa."""

    def __init__(
        self,
        config: Config,
        redeploy: bool = False,
        vespa_url: str = DEFAULT_VESPA_URL,
        vespa_port: int = DEFAULT_VESPA_LOCAL_PORT,
    ):
        self.config = config
        self.schema_name = config.get_schema_name()
        self.app_package_name = config.get_app_package_name()

        rag_params = config.rag_params
        schema_params = config.get_schema_params()

        self.embedding_model_name = rag_params.embedding_model if rag_params else DEFAULT_EMBEDDING_MODEL
        self.embedding_dim = schema_params.get("embedding_dim", 384)
        self.chunk_size = rag_params.chunk_size if rag_params else schema_params.get("chunk_size", 1024)
        self.chunk_overlap = rag_params.chunk_overlap if rag_params else 0
        self.embedding_batch_size = None  # Future enhancement

        logger.info(f"Using model '{self.embedding_model_name}' " f"(dim={self.embedding_dim})")
        self.model = SentenceTransformer(self.embedding_model_name)
        self.app = self._connect_vespa(redeploy, vespa_url, vespa_port, schema_params)

    def feed(self, record: Dict[str, Any]) -> bool:
        """Feed a single record into Vespa.

        Args:
            record: A dictionary with at least 'content' and optional 'loc' fields.

        Returns:
            True if feed succeeded, False otherwise.
        """
        prepared = self._prepare_record(record)
        try:
            response = self.app.feed_data_point(
                schema=self.schema_name,
                data_id=prepared["id"],
                fields=prepared["fields"],
            )
        except Exception as e:
            msg = str(e)
            if "401" in msg and "Unauthorized" in msg:
                logger.error(
                    "Vespa feed returned 401 Unauthorized. "
                    "For Vespa Cloud, set VESPA_CLOUD_SECRET_TOKEN for token auth, "
                    "or VESPA_CLIENT_CERT/VESPA_CLIENT_KEY for mTLS auth."
                )
            logger.error(f"Feed request failed for id={prepared['id']}: {e}")
            return False

        if self._is_success(response):
            logger.success(f"Fed document id={prepared['id']}")
            return True

        logger.error(f"Feed failed for id={prepared['id']}: {getattr(response, 'json', response)}")
        return False

    def _connect_vespa(
        self,
        redeploy: bool,
        vespa_url: str,
        vespa_port: int,
        schema_params: Dict[str, Any],
    ):
        deploy_config = self.config.get_deploy_config()
        cert_path, key_path, ca_cert, verify = get_tls_config_from_deploy(deploy_config)

        # Get cloud secret token for token-based auth (preferred for cloud)
        cloud_token = None
        if deploy_config.is_cloud_mode():
            cloud_token = get_cloud_secret_token(deploy_config)
            if cloud_token:
                logger.info("Using token-based authentication for Vespa Cloud")
            elif cert_path and key_path:
                logger.info("Using mTLS authentication for Vespa Cloud")
            else:
                logger.warning(
                    "No authentication credentials found for Vespa Cloud. "
                    "Set VESPA_CLOUD_SECRET_TOKEN or configure mTLS certificates."
                )

        if redeploy:
            logger.info("Redeploying Vespa application before feeding")
            vespa_schema = VespaSchema(
                schema_name=self.schema_name,
                app_package_name=self.app_package_name,
                **schema_params,
            )
            app_package = vespa_schema.get_package()
            deploy_app_package(None, app_package=app_package, deploy_config=deploy_config)

        logger.info(f"Connecting to Vespa at {vespa_url}:{vespa_port}")
        return make_vespa_client(
            vespa_url,
            vespa_port,
            cert_path,
            key_path,
            ca_cert,
            verify,
            vespa_cloud_secret_token=cloud_token,
        )

    def _prepare_record(self, record: Dict[str, str]) -> Dict[str, Dict[str, object]]:
        content = record.get("content", "").strip()
        if not content:
            raise ValueError("Record is missing content")

        loc = record.get("loc", "").strip()
        data_id = self._make_id(loc)

        chunk_texts = chunks(content, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        logger.info(f"Prepared record id={data_id} with {len(chunk_texts)} chunks ")

        # Batch encode content + chunks to minimize model calls
        embeddings = self._encode_texts([content, *chunk_texts])
        content_embedding, chunk_embeddings = embeddings[0], embeddings[1:]
        if len(chunk_embeddings) != len(chunk_texts):
            raise ValueError(
                "Chunk embeddings count mismatch. " f"Expected {len(chunk_texts)}, got {len(chunk_embeddings)}"
            )

        fields = {
            "id": data_id,
            "loc": loc,
            "content": content,
            "chunks": chunk_texts,
            "chunk_count": len(chunk_texts),
            "content_embedding": self._dense_tensor(content_embedding),
            "chunk_embeddings": self._chunk_tensor(chunk_embeddings),
        }

        return {"id": data_id, "fields": fields}

    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        encode_kwargs: Dict[str, Any] = {
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        if self.embedding_batch_size:
            encode_kwargs["batch_size"] = self.embedding_batch_size
        embeddings = self.model.encode(texts, **encode_kwargs)
        return [vec.tolist() for vec in embeddings]

    def _dense_tensor(self, values: List[float]) -> Dict[str, List[float]]:
        target_dim = self.embedding_dim
        if len(values) != target_dim:
            raise ValueError(f"Tensor length mismatch. Expected {target_dim}, got {len(values)}")
        return {"values": values}

    def _chunk_tensor(self, chunk_vectors: List[List[float]]) -> Dict[str, List[Dict[str, object]]]:
        target_dim = self.embedding_dim
        tensor: Dict[str, List[float]] = {}
        for chunk_idx, vector in enumerate(chunk_vectors):
            if len(vector) != target_dim:
                raise ValueError(
                    f"Chunk {chunk_idx} embedding dim mismatch. " f"Expected {target_dim}, got {len(vector)}"
                )
            tensor[str(chunk_idx)] = vector
        return tensor

    def _make_id(self, loc: str) -> str:
        base = loc if loc else f"nyrag-{uuid.uuid4()}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, base))

    def _is_success(self, response: VespaResponse) -> bool:
        status_code = getattr(response, "status_code", None)
        if status_code is None:
            return False
        if status_code >= 400:
            return False
        return True


def feed_from_config(
    config_path: str,
    record: Dict[str, Any],
    redeploy: bool = False,
) -> bool:
    """Convenience helper to feed a single record using a YAML config path."""
    config = Config.from_yaml(config_path)
    feeder = VespaFeeder(config=config, redeploy=redeploy)
    return feeder.feed(record)
