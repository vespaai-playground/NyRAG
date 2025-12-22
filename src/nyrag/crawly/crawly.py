import io
import logging
import os
import re
from datetime import datetime
from typing import List, Optional, Union

import scrapy
from markitdown import StreamInfo
from markitdown.converters import HtmlConverter
from scrapy import signals
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapy.signalmanager import dispatcher
from scrapy.spiders import CrawlSpider, Rule
from scrapy.utils.log import configure_logging

from nyrag.logger import logger

from .user_agents import get_user_agent


class WebsiteItem(scrapy.Item):
    loc = scrapy.Field()
    content = scrapy.Field()
    title = scrapy.Field()
    timestamp = scrapy.Field()


class Crawly(CrawlSpider):
    name = "crawly"

    def __init__(
        self,
        start_urls: Union[str, List[str]],
        allowed_domains: Optional[List[str]] = None,
        exclude_urls: Optional[List[str]] = None,
        custom_user_agent: Optional[str] = None,
        user_agent_type: str = "chrome",  # New parameter
        respect_robots_txt: bool = True,
        aggressive_crawl: bool = False,
        follow_subdomains: bool = True,
        strict_mode: bool = False,  # New parameter
        processed_urls: Optional[set] = None,  # New parameter for resume
        feeder_callback: Optional[callable] = None,  # New parameter for Vespa feeding
        *args,
        **kwargs,
    ):
        self.processed_urls = processed_urls if processed_urls else set()

        # Handle single URL or list of URLs
        self.start_urls = [start_urls] if isinstance(start_urls, str) else start_urls

        # Set allowed domains if provided, otherwise extract from start_urls
        if allowed_domains:
            self.allowed_domains = allowed_domains
        else:
            if follow_subdomains:
                self.allowed_domains = [self._extract_domain(url) for url in self.start_urls]
            else:
                # Only allow exact domain matches
                self.allowed_domains = [self._extract_exact_domain(url) for url in self.start_urls]

        # Store parameters for rules compilation (BEFORE calling super().__init__)
        self.exclude_urls = exclude_urls if exclude_urls else []
        self.strict_mode = strict_mode

        # Now call parent __init__ which will call _compile_rules
        super(Crawly, self).__init__(*args, **kwargs)
        self.logger.setLevel(logging.ERROR)

        # Configure custom settings
        self.custom_settings = self.custom_settings or {}
        if custom_user_agent:
            self.custom_settings["USER_AGENT"] = custom_user_agent
        else:
            self.custom_settings["USER_AGENT"] = get_user_agent(user_agent_type)

        # Configure robots.txt settings
        if not respect_robots_txt:
            self.custom_settings.update({"ROBOTSTXT_OBEY": False, "COOKIES_ENABLED": False})

        # Configure crawl speed settings
        if aggressive_crawl:
            self.custom_settings.update(
                {
                    "CONCURRENT_REQUESTS": 100,
                    "CONCURRENT_REQUESTS_PER_DOMAIN": 24,
                    "DOWNLOAD_DELAY": 0.1,
                }
            )
        else:
            self.custom_settings.update(
                {
                    "CONCURRENT_REQUESTS": 8,
                    "CONCURRENT_REQUESTS_PER_DOMAIN": 4,
                    "DOWNLOAD_DELAY": 2,
                    "RANDOMIZE_DOWNLOAD_DELAY": True,
                }
            )

        self.items_scraped = 0
        self.html_converter = HtmlConverter()  # Initialize HTML converter
        self.feeder_callback = feeder_callback  # Store feeder callback

        # Connect signals
        dispatcher.connect(self.item_scraped, signals.item_scraped)

    def _compile_rules(self):
        """Override to dynamically compile rules with proper configuration."""
        # Configure URL exclusion patterns
        exclude_patterns = self.exclude_urls if hasattr(self, "exclude_urls") else []

        # Create URL patterns for strict mode
        if self.strict_mode:
            self.start_url_patterns = [self._get_start_url_pattern(url) for url in self.start_urls]
            allow_patterns = self.start_url_patterns
        else:
            allow_patterns = []

        # Set up crawl rules with strict mode patterns
        self.rules = (
            Rule(
                LinkExtractor(
                    allow=allow_patterns,
                    allow_domains=self.allowed_domains,
                    deny=exclude_patterns,
                ),
                callback="parse_page",
                follow=True,
            ),
        )
        super()._compile_rules()

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        return re.findall(r"://([\w\.-]+)/", url + "/")[0]

    def _extract_exact_domain(self, url: str) -> str:
        """Extract base domain without subdomains."""
        domain = self._extract_domain(url)
        parts = domain.split(".")
        # Handle cases like co.uk, com.au etc.
        if len(parts) > 2 and parts[-2] in ["co", "com", "org", "gov", "edu"]:
            return ".".join(parts[-3:])
        return ".".join(parts[-2:])

    def _get_start_url_pattern(self, url: str) -> str:
        """Convert start URL to regex pattern that matches only its subpaths."""
        # Remove protocol and ensure no trailing slash
        cleaned_url = re.sub(r"https?://", "", url.rstrip("/"))
        # Escape special regex characters
        escaped_url = re.escape(cleaned_url)
        # Create pattern that matches exact path and its subpaths
        return rf"https?://{escaped_url}(?:/.*)?$"

    def item_scraped(self, item, response, spider):
        """Update progress when item is scraped"""
        self.items_scraped += 1

    def parse_start_url(self, response):
        """Ensure start_urls are parsed like any other page."""
        return self.parse_page(response)

    def parse_page(self, response):
        """Parse individual pages and extract content."""
        # Skip if already processed (for resume mode)
        if response.url in self.processed_urls:
            return

        logger.info(f"[{self.items_scraped + 1}] Crawling: {response.url}")

        # Convert HTML to markdown using HtmlConverter
        try:
            # Create a BytesIO stream from the response body
            html_stream = io.BytesIO(response.body)

            # Use HtmlConverter directly with UTF-8 charset
            stream_info = StreamInfo(mimetype="text/html", charset="utf-8")
            result = self.html_converter.convert(html_stream, stream_info)

            markdown_content = result.markdown.strip()

            # Use MarkItDown's extracted title (e.g., from HTML <title> tag)
            # Fall back to first markdown heading, then URL
            if result.title:
                title = result.title
            else:
                title_match = re.match(r"^#\s+(.+)$", markdown_content, re.MULTILINE)
                title = title_match.group(1) if title_match else response.url

        except Exception as e:
            logger.error(f"Error converting {response.url} to markdown: {e}")
            markdown_content = ""
            title = response.url

        item = WebsiteItem()
        item["loc"] = response.url
        item["content"] = markdown_content
        item["title"] = title
        item["timestamp"] = datetime.utcnow().isoformat()

        yield item

        # Feed to Vespa if feeder callback is provided
        if self.feeder_callback and markdown_content:
            try:
                self.feeder_callback(
                    {
                        "loc": response.url,
                        "content": markdown_content,
                        "title": title,
                    }
                )
            except Exception as e:
                logger.error(f"Failed to feed {response.url} to Vespa: {e}")


def crawl_web(
    start_urls,
    output_dir,
    allowed_domains=None,
    exclude_urls=None,
    output_file="output.jsonl",
    respect_robots_txt=True,
    aggressive_crawl=False,
    follow_subdomains=True,
    strict_mode=False,  # New parameter
    user_agent_type="chrome",  # New parameter
    custom_user_agent=None,  # New parameter
    resume=False,  # New parameter for resume
    processed_urls=None,  # New parameter for resume
    feeder_callback=None,  # New parameter for Vespa feeding
):
    configure_logging(install_root_handler=False)

    logging.basicConfig(level=logging.WARNING)

    logging.getLogger("scrapy").setLevel(logging.ERROR)
    logging.getLogger("scrapy.core.engine").setLevel(logging.ERROR)
    logging.getLogger("scrapy.core.scraper").setLevel(logging.ERROR)
    logging.getLogger("scrapy.downloadermiddlewares.redirect").setLevel(logging.ERROR)
    logging.getLogger("twisted").setLevel(logging.ERROR)

    logging.getLogger("charset_normalizer").setLevel(logging.WARNING)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct full output path
    output_path = os.path.join(output_dir, output_file)

    # Decide which UA to use
    if custom_user_agent:
        ua = custom_user_agent
    else:
        ua = get_user_agent(user_agent_type)

    process = CrawlerProcess(
        settings={
            "FEEDS": {
                output_path: {
                    "format": "jsonlines",
                    "encoding": "utf8",
                    "overwrite": not resume,  # Don't overwrite if resuming
                }
            },
            "LOG_ENABLED": False,
            "LOG_LEVEL": "ERROR",
            "ROBOTSTXT_OBEY": respect_robots_txt,
            # Add stats collection
            "STATS_CLASS": "scrapy.statscollectors.MemoryStatsCollector",
            "DOWNLOADER_MIDDLEWARES": {
                "scrapy.downloadermiddlewares.retry.RetryMiddleware": 550,
            },
            "DEFAULT_REQUEST_HEADERS": {
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,"
                    "image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
            },
            "RETRY_HTTP_CODES": [500, 502, 503, 504, 522, 524, 408, 429, 403],
            "RETRY_TIMES": 3,
            "HTTPERROR_ALLOWED_CODES": [],
            "USER_AGENT": ua,
        }
    )

    process.crawl(
        Crawly,
        start_urls=start_urls,
        allowed_domains=allowed_domains,
        exclude_urls=exclude_urls,
        respect_robots_txt=respect_robots_txt,
        aggressive_crawl=aggressive_crawl,
        follow_subdomains=follow_subdomains,
        strict_mode=strict_mode,  # Pass strict_mode parameter
        user_agent_type=user_agent_type,
        custom_user_agent=custom_user_agent,
        processed_urls=(processed_urls if resume else set()),  # Pass processed URLs if resuming
        feeder_callback=feeder_callback,  # Pass feeder callback
    )
    process.start()
