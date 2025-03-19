import click
from r2r_scrapy.cli.commands.scrape_command import scrape
from r2r_scrapy.cli.commands.list_collections_command import list_collections
from r2r_scrapy.cli.commands.create_collection_command import create_collection
from r2r_scrapy.cli.commands.delete_collection_command import delete_collection
from r2r_scrapy.cli.commands.list_documents_command import list_documents
from r2r_scrapy.cli.commands.get_document_command import get_document
from r2r_scrapy.cli.commands.delete_document_command import delete_document
from r2r_scrapy.cli.commands.generate_report_command import generate_report

@click.group()
@click.version_option(version='1.0.0')
def cli():
    """R2R Scrapy - Asynchronous Documentation Collector for RAG Systems"""
    pass

# Add commands
cli.add_command(scrape)
cli.add_command(list_collections)
cli.add_command(create_collection)
cli.add_command(delete_collection)
cli.add_command(list_documents)
cli.add_command(get_document)
cli.add_command(delete_document)
cli.add_command(generate_report)

if __name__ == '__main__':
    cli() 