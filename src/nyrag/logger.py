"""Custom logger for nyrag using rich library."""

import logging
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape as rich_escape
from rich.theme import Theme


# Custom theme for better looking logs
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "critical": "bold white on red",
        "success": "bold green",
        "debug": "dim cyan",
    }
)

# Create a rich console with custom theme
console = Console(theme=custom_theme, stderr=True)


class NyragLogger:
    """Custom logger wrapper with rich formatting."""

    def __init__(self, name: str = "nyrag", level: str = "INFO"):
        """Initialize the logger.

        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Remove existing handlers
        self.logger.handlers = []

        # Create rich handler
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            omit_repeated_times=False,
        )
        rich_handler.setFormatter(
            logging.Formatter(
                "%(message)s",
                datefmt="[%X]",
            )
        )

        self.logger.addHandler(rich_handler)
        self.logger.propagate = False

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        safe = rich_escape(str(message))
        self.logger.debug(f"[debug]{safe}[/debug]", extra={"markup": True}, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        safe = rich_escape(str(message))
        self.logger.info(f"[info]{safe}[/info]", extra={"markup": True}, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        safe = rich_escape(str(message))
        self.logger.warning(f"[warning]{safe}[/warning]", extra={"markup": True}, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        safe = rich_escape(str(message))
        self.logger.error(f"[error]{safe}[/error]", extra={"markup": True}, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        safe = rich_escape(str(message))
        self.logger.critical(f"[critical]{safe}[/critical]", extra={"markup": True}, **kwargs)

    def success(self, message: str, **kwargs):
        """Log success message."""
        safe = rich_escape(str(message))
        self.logger.info(f"[success]✓ {safe}[/success]", extra={"markup": True}, **kwargs)

    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        safe = rich_escape(str(message))
        self.logger.exception(f"[error]{safe}[/error]", extra={"markup": True}, **kwargs)


# Global logger instance
logger = NyragLogger()


def set_log_level(level: str):
    """Set the global logger level.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger.logger.setLevel(getattr(logging, level.upper()))


def get_logger(name: Optional[str] = None) -> NyragLogger:
    """Get a logger instance.

    Args:
        name: Optional logger name. If None, returns the global logger.

    Returns:
        NyragLogger instance
    """
    if name is None:
        return logger
    return NyragLogger(name)
