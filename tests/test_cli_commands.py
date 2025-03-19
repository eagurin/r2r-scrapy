import pytest
from click.testing import CliRunner
import tempfile
import os
import json
from r2r_scrapy.cli.commands.scrape_command import scrape
from r2r_scrapy.cli.commands.list_collections_command import list_collections
from r2r_scrapy.cli.commands.create_collection_command import create_collection
from r2r_scrapy.cli.commands.delete_collection_command import delete_collection
from r2r_scrapy.cli.commands.list_documents_command import list_documents
from r2r_scrapy.cli.commands.get_document_command import get_document
from r2r_scrapy.cli.commands.delete_document_command import delete_document
from r2r_scrapy.cli.commands.generate_report_command import generate_report

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def config_file():
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        config = {
            'r2r': {
                'api_url': 'http://test.com',
                'api_key': 'test_key',
                'batch_size': 10
            }
        }
        json.dump(config, f)
        return f.name

def test_scrape_command(runner, config_file):
    result = runner.invoke(scrape, [
        '--library', 'test_lib',
        '--url', 'http://test.com',
        '--config', config_file
    ])
    assert result.exit_code == 0

def test_list_collections_command(runner, config_file):
    result = runner.invoke(list_collections, [
        '--format', 'json',
        '--config', config_file
    ])
    assert result.exit_code == 0

def test_create_collection_command(runner, config_file):
    result = runner.invoke(create_collection, [
        '--name', 'test_collection',
        '--description', 'Test description',
        '--config', config_file
    ])
    assert result.exit_code == 0

def test_delete_collection_command(runner, config_file):
    result = runner.invoke(delete_collection, [
        '--id', 'test_collection',
        '--force',
        '--config', config_file
    ])
    assert result.exit_code == 0

def test_list_documents_command(runner, config_file):
    result = runner.invoke(list_documents, [
        '--collection', 'test_collection',
        '--format', 'json',
        '--config', config_file
    ])
    assert result.exit_code == 0

def test_get_document_command(runner, config_file):
    result = runner.invoke(get_document, [
        '--id', 'test_doc',
        '--format', 'json',
        '--config', config_file
    ])
    assert result.exit_code == 0

def test_delete_document_command(runner, config_file):
    result = runner.invoke(delete_document, [
        '--id', 'test_doc',
        '--force',
        '--config', config_file
    ])
    assert result.exit_code == 0

def test_generate_report_command(runner, config_file):
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
        result = runner.invoke(generate_report, [
            '--format', 'html',
            '--output', f.name,
            '--config', config_file
        ])
        assert result.exit_code == 0
        assert os.path.exists(f.name)
        os.unlink(f.name) 