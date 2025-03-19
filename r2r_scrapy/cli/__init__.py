"""
R2R Scrapy Command Line Interface
"""

from r2r_scrapy.cli.main import cli
from r2r_scrapy.cli.commands.base_command import BaseCommand

__all__ = [
    'cli',
    'BaseCommand',
]
