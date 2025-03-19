"""
R2R Scrapy External Integrations
"""

from r2r_scrapy.integrations.github_integration import GitHubIntegration
from r2r_scrapy.integrations.stackoverflow_integration import StackOverflowIntegration
from r2r_scrapy.integrations.wikipedia_integration import WikipediaIntegration

__all__ = [
    'GitHubIntegration',
    'StackOverflowIntegration',
    'WikipediaIntegration',
]
