"""
R2R Scrapy Content Processors
"""

from r2r_scrapy.processors.code_processor import CodeProcessor
from r2r_scrapy.processors.markdown_processor import MarkdownProcessor
from r2r_scrapy.processors.api_processor import APIDocProcessor
from r2r_scrapy.processors.html_processor import HTMLProcessor

__all__ = [
    'CodeProcessor',
    'MarkdownProcessor',
    'APIDocProcessor',
    'HTMLProcessor',
]
