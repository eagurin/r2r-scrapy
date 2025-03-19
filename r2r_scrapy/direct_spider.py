#!/usr/bin/env python
"""
Прямой запуск паука без использования механизма имен Scrapy
"""

import asyncio
import logging
import sys
from datetime import datetime

import click
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from r2r_scrapy.exporters.r2r_exporter import R2RExporter
from r2r_scrapy.spiders.api_doc_spider import ApiDocSpider

# Настройка логгирования
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


@click.command()
@click.option("--library", required=True, help="Library name to scrape")
@click.option("--url", required=True, help="URL to start scraping from")
@click.option(
    "--chunking", default="semantic", help="Chunking strategy to use"
)
@click.option(
    "--allowed-paths", help="Comma-separated list of allowed URL paths"
)
@click.option(
    "--follow-links", is_flag=True, default=True, help="Follow links on pages"
)
@click.option(
    "--max-pages", default=5000, help="Maximum number of pages to crawl"
)
def run_spider(library, url, chunking, allowed_paths, follow_links, max_pages):
    """Запуск паука напрямую"""
    logger.info(f"Starting direct spider run for {library} at {url}")

    # Получаем настройки проекта
    settings = get_project_settings()
    settings.update(
        {
            "LIBRARY_NAME": library,
            "CHUNKING_STRATEGY": chunking,
            "LOG_LEVEL": "DEBUG",
            "DEPTH_LIMIT": 3,  # Ограничение глубины обхода
            "DEPTH_PRIORITY": 1,  # Приоритет глубины (положительное значение для BFS)
            "SCHEDULER_DISK_QUEUE": "scrapy.squeues.PickleFifoDiskQueue",
            "SCHEDULER_MEMORY_QUEUE": "scrapy.squeues.FifoMemoryQueue",
            "CLOSESPIDER_PAGECOUNT": max_pages,  # Остановка после обхода max_pages страниц
        }
    )

    # Создаем процесс Crawler
    process = CrawlerProcess(settings)

    # Напрямую создаем экземпляр паука
    spider_args = {
        "url": url,
        "library": library,
        "allowed_paths": allowed_paths,
        "follow_links": follow_links,  # Явно передаем параметр follow_links
    }

    # Запускаем паук напрямую по классу, а не по имени
    process.crawl(ApiDocSpider, **spider_args)
    process.start()

    logger.info("Spider run completed")


if __name__ == "__main__":
    run_spider()

    # Выход с кодом успеха
    sys.exit(0)
