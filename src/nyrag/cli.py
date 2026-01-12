import argparse
import os
import sys

from nyrag.config import Config
from nyrag.logger import logger
from nyrag.process import process_from_config
from nyrag.vespa_cli import set_vespa_target_cloud, vespa_auth_login


def cmd_process(args):
    """Process documents based on config."""
    try:
        # Load configuration
        config = Config.from_yaml(args.config)

        logger.info(f"Project: {config.name}")
        logger.info(f"Mode: {config.mode}")
        logger.info(f"Output directory: {config.get_output_path()}")

        if args.resume:
            logger.info("Resume mode enabled - will skip already processed items")

        logger.info("Vespa feeding enabled - documents will be fed to Vespa as they are processed")

        # Process based on config
        process_from_config(config, resume=args.resume, config_path=args.config)

        logger.success(f"Processing complete! Output saved to {config.get_output_path()}")

    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        sys.exit(1)


def cmd_ui(args):
    """Start the UI server."""
    try:
        import uvicorn

        # Handle cloud mode initialization
        if args.cloud:
            logger.info("Cloud mode selected. Setting Vespa target to cloud...")
            if not set_vespa_target_cloud():
                logger.error("Failed to set Vespa target to cloud. Is Vespa CLI installed?")
                sys.exit(1)

            logger.info("Initiating Vespa Cloud authentication...")
            logger.info("Please complete the login in your browser to continue.")

            if not vespa_auth_login():
                logger.error("Vespa Cloud authentication failed or was cancelled.")
                sys.exit(1)

            logger.success("Vespa Cloud authentication successful!")
            # Set environment variable so API knows we're in cloud mode
            os.environ["NYRAG_CLOUD_MODE"] = "1"

        from nyrag.api import app

        # Also set state on the app object to be sure
        if args.cloud:
            app.state.cloud_mode = True

        logger.info(f"Starting UI server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    except ImportError:
        logger.error("uvicorn is required to run the UI. Install it with: pip install uvicorn")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting UI server: {str(e)}")
        sys.exit(1)


def main():
    """Main CLI entry point for nyrag."""
    parser = argparse.ArgumentParser(
        description="nyrag - Web crawler and document processor for RAG applications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command (existing functionality)
    process_parser = subparsers.add_parser(
        "process",
        help="Process documents from URLs or files",
    )
    process_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    process_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing crawl/processing data (skip already processed URLs/files)",
    )
    process_parser.set_defaults(func=cmd_process)

    # UI command (new)
    ui_parser = subparsers.add_parser(
        "ui",
        help="Start the web UI server",
    )
    ui_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    ui_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)",
    )
    ui_parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Logging level (default: info)",
    )
    ui_parser.add_argument(
        "--cloud",
        action="store_true",
        help="Start in cloud mode (will prompt for Vespa Cloud authentication)",
    )
    ui_parser.set_defaults(func=cmd_ui)

    args = parser.parse_args()

    # If no command specified, show help
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    # Execute the command
    args.func(args)


if __name__ == "__main__":
    main()
