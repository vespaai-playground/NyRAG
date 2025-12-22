import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

from markitdown import MarkItDown, StreamInfo

from nyrag.config import Config
from nyrag.crawly import crawl_web
from nyrag.deploy import deploy_app_package
from nyrag.feed import VespaFeeder
from nyrag.logger import logger
from nyrag.schema import VespaSchema
from nyrag.utils import resolve_vespa_port


def load_processed_locations(jsonl_file: Path) -> set:
    """
    Load already processed locations from a JSONL file.

    Args:
        jsonl_file: Path to the JSONL file

    Returns:
        Set of processed locations (URLs or file paths)
    """
    processed = set()

    if not jsonl_file.exists():
        return processed

    try:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if "loc" in data:
                        processed.add(data["loc"])
        logger.info(f"Loaded {len(processed)} already processed items from {jsonl_file}")
    except Exception as e:
        logger.warning(f"Failed to load processed items from {jsonl_file}: {e}")

    return processed


def save_to_jsonl(
    loc: str,
    content: str,
    title: str,
    output_path: Path,
    output_file: str = "data.jsonl",
):
    """
    Save data to JSONL file in unified format.

    Args:
        loc: Location (URL or file path)
        content: Markdown content
        title: Document title
        output_path: Directory to save the file
        output_file: Name of the JSONL file
    """
    # Save directly to output path
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / output_file

    # Append to JSONL file
    with open(file_path, "a", encoding="utf-8") as f:
        json_line = json.dumps(
            {
                "loc": loc,
                "content": content,
                "title": title,
                "timestamp": datetime.utcnow().isoformat(),
            },
            ensure_ascii=False,
        )
        f.write(json_line + "\n")

    logger.debug(f"Saved to {file_path}")


def process_from_config(config: Config, resume: bool = False):
    """
    Process based on configuration file.

    Args:
        config: Configuration object
        resume: If True, skip already processed URLs/files
    """
    output_dir = config.get_output_path()

    # Check if project already exists
    existing_file = output_dir / "data.jsonl"

    if existing_file.exists() and not resume:
        logger.error(f"Project '{config.name}' already exists at {output_dir}")
        logger.error(f"Found existing data file: {existing_file}")
        logger.error(
            "Use --resume flag to continue with the existing project, or remove the project directory to start fresh"
        )
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Create and save Vespa schema
    _create_schema(config)

    # Step 2: Process data based on mode
    if config.is_web_mode():
        _process_web(config, output_dir, resume=resume)
    elif config.is_docs_mode():
        _process_documents(config, output_dir, resume=resume)
    else:
        raise ValueError(f"Unknown mode: {config.mode}")


def _create_schema(config: Config):
    """
    Create and save Vespa schema.

    Args:
        config: Configuration object
    """
    logger.info("Creating Vespa schema...")

    schema_params = config.get_schema_params()
    schema_name = config.get_schema_name()
    app_package_name = config.get_app_package_name()
    app_path = config.get_app_path()

    logger.info(f"Schema name: {schema_name}")
    logger.info(f"App package name: {app_package_name}")
    logger.info(f"Schema parameters: {schema_params}")

    vespa_schema = VespaSchema(schema_name=schema_name, app_package_name=app_package_name, **schema_params)

    app_package = vespa_schema.get_package()
    app_package.to_files(str(app_path))
    logger.success(f"Schema saved to {app_path}")

    deploy_enabled = os.getenv("NYRAG_VESPA_DEPLOY", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if not deploy_enabled:
        logger.warning("NYRAG_VESPA_DEPLOY disabled; skipping Vespa application deploy")
        return

    deployed = deploy_app_package(app_path, app_package=app_package)
    if not deployed:
        logger.error("Vespa deploy failed; aborting.")
        raise SystemExit(1)


def _process_web(config: Config, output_dir: Path, resume: bool = False):
    """
    Process web URLs: crawl and convert to markdown.

    Args:
        config: Configuration object
        output_dir: Directory to save output
        resume: If True, skip already crawled URLs
    """

    urls = [config.start_loc]
    exclude_urls = config.exclude or []
    crawl_params = config.crawl_params

    logger.info(f"Starting web crawl from: {config.start_loc}")
    if exclude_urls:
        logger.info(f"Excluding: {exclude_urls}")

    # Log crawl parameters
    logger.info("Crawl settings:")
    logger.info(f"  - Respect robots.txt: {crawl_params.respect_robots_txt}")
    logger.info(f"  - Aggressive crawl: {crawl_params.aggressive_crawl}")
    logger.info(f"  - Follow subdomains: {crawl_params.follow_subdomains}")
    logger.info(f"  - Strict mode: {crawl_params.strict_mode}")
    logger.info(f"  - User agent: {crawl_params.user_agent_type}")
    logger.info("  - Feed to Vespa: enabled")

    # Load already crawled URLs if resuming
    processed_urls = set()
    if resume:
        jsonl_file = output_dir / "data.jsonl"
        processed_urls = load_processed_locations(jsonl_file)

    # Initialize Vespa feeder
    logger.info("Initializing Vespa feeder...")
    vespa_url = (os.getenv("VESPA_URL") or "").strip() or "http://localhost"
    vespa_port = resolve_vespa_port(vespa_url)
    feeder = VespaFeeder(config=config, redeploy=False, vespa_url=vespa_url, vespa_port=vespa_port)
    feeder_callback = feeder.feed

    try:
        crawl_web(
            urls,
            output_dir=output_dir,
            allowed_domains=crawl_params.allowed_domains,
            exclude_urls=exclude_urls,
            output_file="data.jsonl",
            respect_robots_txt=crawl_params.respect_robots_txt,
            aggressive_crawl=crawl_params.aggressive_crawl,
            follow_subdomains=crawl_params.follow_subdomains,
            strict_mode=crawl_params.strict_mode,
            user_agent_type=crawl_params.user_agent_type,
            custom_user_agent=crawl_params.custom_user_agent,
            resume=resume,
            processed_urls=processed_urls,
            feeder_callback=feeder_callback,
        )
        logger.success("Web crawling completed")
    except Exception as e:
        logger.error(f"Crawling failed: {str(e)}")
        raise


def _process_documents(config: Config, output_dir: Path, resume: bool = False):
    """
    Process documents: convert to markdown.

    Args:
        config: Configuration object
        output_dir: Directory to save output
        resume: If True, skip already processed files
    """
    start_loc = Path(config.start_loc)
    exclude_patterns = config.exclude or []
    doc_params = config.doc_params

    logger.info("Document processing settings:")
    logger.info(f"  - Recursive: {doc_params.recursive}")
    logger.info(f"  - Include hidden: {doc_params.include_hidden}")
    logger.info(f"  - Follow symlinks: {doc_params.follow_symlinks}")
    if doc_params.max_file_size_mb:
        logger.info(f"  - Max file size: {doc_params.max_file_size_mb} MB")
    if doc_params.file_extensions:
        logger.info(f"  - File extensions: {', '.join(doc_params.file_extensions)}")
    logger.info("  - Feed to Vespa: enabled")

    # Check if start_loc is a directory or file
    if start_loc.is_dir():
        logger.info(f"Processing documents from directory: {start_loc}")
        documents = _collect_documents(start_loc, doc_params)
    elif start_loc.is_file():
        logger.info(f"Processing single document: {start_loc}")
        documents = [start_loc]
    else:
        logger.error(f"Invalid start_loc: {start_loc}")
        return

    # Apply exclusions
    if exclude_patterns:
        original_count = len(documents)
        documents = _apply_exclusions(documents, exclude_patterns)
        logger.info(f"Excluded {original_count - len(documents)} files based on patterns")

    # Apply file extension filter
    if doc_params.file_extensions:
        original_count = len(documents)
        documents = [
            d
            for d in documents
            if d.suffix.lower()
            in [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in doc_params.file_extensions]
        ]
        logger.info(f"Filtered to {len(documents)} files by extension (excluded {original_count - len(documents)})")

    # Apply file size filter
    if doc_params.max_file_size_mb:
        original_count = len(documents)
        max_bytes = doc_params.max_file_size_mb * 1024 * 1024
        documents = [d for d in documents if d.stat().st_size <= max_bytes]
        logger.info(f"Filtered to {len(documents)} files by size (excluded {original_count - len(documents)})")

    if not documents:
        logger.error("No documents found to process")
        return

    logger.info(f"Processing {len(documents)} document(s)")

    # Setup output file
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_file = output_dir / "data.jsonl"

    # Load already processed files if resuming
    processed_files = set()
    if resume:
        processed_files = load_processed_locations(jsonl_file)
        # Filter out already processed documents
        original_count = len(documents)
        documents = [d for d in documents if str(d.absolute()) not in processed_files]
        skipped_count = original_count - len(documents)
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} already processed files")
        if not documents:
            logger.success("All documents have already been processed")
            return
    else:
        # Clear existing JSONL file if not resuming
        if jsonl_file.exists():
            jsonl_file.unlink()

    # Initialize Vespa feeder
    logger.info("Initializing Vespa feeder...")
    vespa_url = (os.getenv("VESPA_URL") or "").strip() or "http://localhost"
    vespa_port = resolve_vespa_port(vespa_url)
    feeder = VespaFeeder(config=config, redeploy=False, vespa_url=vespa_url, vespa_port=vespa_port)

    md = MarkItDown()
    success_count = 0
    error_count = 0

    for doc_file in documents:
        try:
            logger.info(f"Processing: {doc_file.name}")

            # Try to convert the file
            try:
                result = md.convert(str(doc_file))
            except Exception as e:
                # Check if it's an encoding error, either directly or wrapped
                if isinstance(e, UnicodeDecodeError) or "UnicodeDecodeError" in str(e):
                    # If we get a UnicodeDecodeError, retry with explicit UTF-8 encoding hint
                    logger.warning(f"Encoding error for {doc_file.name}, retrying with UTF-8 encoding")
                    stream_info = StreamInfo(charset="utf-8")
                    result = md.convert(str(doc_file), stream_info=stream_info)
                else:
                    raise e

            # Use absolute path as loc
            loc = str(doc_file.absolute())

            # Use MarkItDown's extracted title, fall back to filename
            title = result.title if result.title else doc_file.stem

            save_to_jsonl(loc, result.markdown, title, output_dir)
            success_count += 1

            # Feed to Vespa if requested
            if feeder and result.markdown:
                try:
                    feeder.feed(
                        {
                            "loc": loc,
                            "content": result.markdown,
                            "title": title,
                        }
                    )
                except Exception as e:
                    logger.error(f"Failed to feed {doc_file.name} to Vespa: {e}")

        except Exception as e:
            logger.error(f"Failed to process {doc_file}: {str(e)}")
            error_count += 1

    logger.success(f"Document processing completed: {success_count} successful, {error_count} failed")


def _collect_documents(directory: Path, doc_params) -> List[Path]:
    """
    Collect documents from directory based on parameters.

    Args:
        directory: Directory to collect from
        doc_params: Document processing parameters

    Returns:
        List of document paths
    """
    documents = []

    if doc_params.recursive:
        pattern = "**/*"
    else:
        pattern = "*"

    for item in directory.glob(pattern):
        # Skip if not a file
        if not item.is_file():
            continue

        # Skip hidden files unless explicitly included
        if not doc_params.include_hidden and item.name.startswith("."):
            continue

        # Handle symlinks
        if item.is_symlink() and not doc_params.follow_symlinks:
            continue

        documents.append(item)

    return documents


def _apply_exclusions(documents: List[Path], exclude_patterns: List[str]) -> List[Path]:
    """
    Apply exclusion patterns to document list.

    Args:
        documents: List of document paths
        exclude_patterns: List of exclusion patterns

    Returns:
        Filtered list of documents
    """
    filtered = []

    for doc in documents:
        excluded = False
        for pattern in exclude_patterns:
            # Handle wildcard patterns
            if pattern.startswith("*"):
                if str(doc).endswith(pattern.lstrip("*")):
                    excluded = True
                    break
            # Handle full path patterns
            elif pattern in str(doc):
                excluded = True
                break
            # Handle filename patterns
            elif pattern == doc.name:
                excluded = True
                break

        if not excluded:
            filtered.append(doc)

    return filtered
