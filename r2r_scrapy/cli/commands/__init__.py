"""
R2R Scrapy CLI Commands
"""

from r2r_scrapy.cli.commands.scrape_command import scrape
from r2r_scrapy.cli.commands.list_collections_command import list_collections
from r2r_scrapy.cli.commands.create_collection_command import create_collection
from r2r_scrapy.cli.commands.delete_collection_command import delete_collection
from r2r_scrapy.cli.commands.list_documents_command import list_documents
from r2r_scrapy.cli.commands.get_document_command import get_document
from r2r_scrapy.cli.commands.delete_document_command import delete_document
from r2r_scrapy.cli.commands.generate_report_command import generate_report

__all__ = [
    'scrape',
    'list_collections',
    'create_collection',
    'delete_collection',
    'list_documents',
    'get_document',
    'delete_document',
    'generate_report',
]
