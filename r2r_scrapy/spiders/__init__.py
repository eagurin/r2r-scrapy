"""
R2R Scrapy Spiders
"""

from r2r_scrapy.spiders.api_spider import APIDocSpider
from r2r_scrapy.spiders.tutorial_spider import TutorialSpider
from r2r_scrapy.spiders.github_spider import GitHubSpider
from r2r_scrapy.spiders.blog_spider import BlogSpider

__all__ = [
    'APIDocSpider',
    'TutorialSpider',
    'GitHubSpider',
    'BlogSpider',
]
