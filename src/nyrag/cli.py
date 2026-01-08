import argparse
import sys

from nyrag.config import Config
from nyrag.logger import logger
from nyrag.process import process_from_config


def main():
    """Main CLI entry point for nyrag."""
    parser = argparse.ArgumentParser(
        description="nyrag - Web crawler and document processor for RAG applications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing crawl/processing data (skip already processed URLs/files)",
    )

    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
