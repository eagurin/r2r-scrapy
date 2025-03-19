Assistant
Необходимые импорты для всех файлов R2R Scrapy

Spiders

API Spider (spiders/api_spider.py)

import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
import re
from r2r_scrapy.processors.api_processor import APIDocProcessor
Tutorial Spider (spiders/tutorial_spider.py)

import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
import re
from r2r_scrapy.processors.markdown_processor import MarkdownProcessor
from r2r_scrapy.processors.html_processor import HTMLProcessor
GitHub Spider (spiders/github_spider.py)

import scrapy
import json
import base64
import re
from urllib.parse import urljoin
from r2r_scrapy.processors.markdown_processor import MarkdownProcessor
from r2r_scrapy.processors.code_processor import CodeProcessor
Blog Spider (spiders/blog_spider.py)

import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from datetime import datetime
import re
from r2r_scrapy.processors.html_processor import HTMLProcessor
Processors

Code Processor (processors/code_processor.py)

import re
from pygments import lexers, highlight
from pygments.formatters import HtmlFormatter
Markdown Processor (processors/markdown_processor.py)

import re
import markdown
from bs4 import BeautifulSoup
import html2text
API Processor (processors/api_processor.py)

import re
from bs4 import BeautifulSoup
from r2r_scrapy.processors.code_processor import CodeProcessor
HTML Processor (processors/html_processor.py)

import re
from bs4 import BeautifulSoup
import html2text
from r2r_scrapy.processors.code_processor import CodeProcessor
Chunkers

Semantic Chunker (chunkers/semantic_chunker.py)

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
Code Chunker (chunkers/code_chunker.py)

import re
Markdown Chunker (chunkers/markdown_chunker.py)

import re
Recursive Chunker (chunkers/recursive_chunker.py)

import re
Exporters

R2R Exporter (exporters/r2r_exporter.py)

import asyncio
import aiohttp
import json
import hashlib
import uuid
from datetime import datetime
File Exporter (exporters/file_exporter.py)

import os
import json
import yaml
import hashlib
from datetime import datetime
Middleware

JavaScript Middleware (middleware/javascript_middleware.py)

from scrapy.http import HtmlResponse
from scrapy.downloadermiddlewares.retry import RetryMiddleware
import logging
import asyncio
from playwright.async_api import async_playwright
Rate Limiter (middleware/rate_limiter.py)

import time
import logging
import random
from scrapy.exceptions import IgnoreRequest
from collections import defaultdict
Utils

URL Prioritizer (utils/url_prioritizer.py)

import re
from urllib.parse import urlparse
import logging
Resource Manager (utils/resource_manager.py)

import psutil
import time
import logging
import threading
import asyncio
from collections import deque
Quality Monitor (utils/quality_monitor.py)

import logging
import re
import time
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
Version Control (utils/version_control.py)

import os
import re
import git
import logging
import hashlib
import json
from datetime import datetime
Pipelines

Preprocessing Pipeline (pipelines/preprocessing_pipeline.py)

import logging
from bs4 import BeautifulSoup
import re
import html2text
Content Pipeline (pipelines/content_pipeline.py)

import logging
import re
from r2r_scrapy.processors.code_processor import CodeProcessor
from r2r_scrapy.processors.markdown_processor import MarkdownProcessor
from r2r_scrapy.processors.html_processor import HTMLProcessor
from r2r_scrapy.processors.api_processor import APIDocProcessor
Chunking Pipeline (pipelines/chunking_pipeline.py)

import logging
from r2r_scrapy.chunkers.semantic_chunker import SemanticChunker
from r2r_scrapy.chunkers.code_chunker import CodeChunker
from r2r_scrapy.chunkers.markdown_chunker import MarkdownChunker
from r2r_scrapy.chunkers.recursive_chunker import RecursiveChunker
R2R Pipeline (pipelines/r2r_pipeline.py)

import logging
import asyncio
from r2r_scrapy.exporters.r2r_exporter import R2RExporter
Integrations

GitHub Integration (integrations/github_integration.py)

import aiohttp
import asyncio
import base64
import json
import logging
from urllib.parse import quote
Stack Overflow Integration (integrations/stackoverflow_integration.py)

import aiohttp
import asyncio
import logging
import html
import re
from urllib.parse import quote
Wikipedia Integration (integrations/wikipedia_integration.py)

import aiohttp
import asyncio
import logging
import re
from urllib.parse import quote
Security

Key Manager (security/key_manager.py)

import os
import json
import base64
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
Secure Logger (security/secure_logger.py)

import logging
import re
import json
import os
from datetime import datetime
CLI Commands

Base Command (cli/commands/base_command.py)

import click
import logging
import os
from r2r_scrapy.config import Config
Scrape Command (cli/commands/scrape_command.py)

import click
import asyncio
import os
import logging
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter
List Collections Command (cli/commands/list_collections_command.py)

import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter
Create Collection Command (cli/commands/create_collection_command.py)

import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter
Delete Collection Command (cli/commands/delete_collection_command.py)

import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter
List Documents Command (cli/commands/list_documents_command.py)

import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter
Get Document Command (cli/commands/get_document_command.py)

import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter
Delete Document Command (cli/commands/delete_document_command.py)

import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter
Generate Report Command (cli/commands/generate_report_command.py)

import click
import asyncio
import os
import logging
import json
from datetime import datetime
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter
Основные файлы

Config (config.py)

import os
import yaml
from typing import Dict, Any, Optional
Settings (settings.py)

import os
from r2r_scrapy.config import Config
Main Entry Point (main.py)

import sys
import os
from r2r_scrapy.cli.main import main
import asyncio
CLI Main (cli/main.py)

import click
from r2r_scrapy.cli.commands.scrape_command import scrape
from r2r_scrapy.cli.commands.list_collections_command import list_collections
from r2r_scrapy.cli.commands.create_collection_command import create_collection
from r2r_scrapy.cli.commands.delete_collection_command import delete_collection
from r2r_scrapy.cli.commands.list_documents_command import list_documents
from r2r_scrapy.cli.commands.get_document_command import get_document
from r2r_scrapy.cli.commands.delete_document_command import delete_document
from r2r_scrapy.cli.commands.generate_report_command import generate_report
Setup (setup.py)

from setuptools import setup, find_packages
