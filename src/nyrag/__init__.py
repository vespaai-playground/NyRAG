__version__ = "0.0.3"

import logging

from nyrag.logger import get_logger, logger, set_log_level


logging.getLogger("scrapy").propagate = False

__all__ = ["logger", "get_logger", "set_log_level"]
