import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from r2r_scrapy.cli.commands.create_collection_command import create_collection
from r2r_scrapy.cli.commands.delete_collection_command import delete_collection
from r2r_scrapy.cli.commands.delete_document_command import delete_document
from r2r_scrapy.cli.commands.generate_report_command import generate_report
from r2r_scrapy.cli.commands.get_document_command import get_document
from r2r_scrapy.cli.commands.list_collections_command import list_collections
from r2r_scrapy.cli.commands.list_documents_command import list_documents
from r2r_scrapy.cli.commands.scrape_command import scrape


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def config_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        config = {
            "r2r": {
                "api_url": "http://test.com",
                "api_key": "test_key",
                "batch_size": 10,
            }
        }
        json.dump(config, f)
        return f.name


# Мокаем функцию процесса Scrapy, чтобы предотвратить запуск реального паука
@pytest.fixture
def mock_process():
    with patch("scrapy.crawler.CrawlerProcess.start") as mock_start:
        yield mock_start


# Мокаем функцию run_until_complete, чтобы предотвратить реальные асинхронные вызовы
@pytest.fixture
def mock_loop():
    with patch("asyncio.get_event_loop") as mock_get_loop:
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = {
            "id": "test_collection_id"
        }
        mock_get_loop.return_value = mock_loop
        yield mock_loop


@patch("asyncio.get_event_loop")
def test_scrape_command(mock_get_loop, runner, config_file, mock_process):
    # Настраиваем мок для event loop
    mock_loop = MagicMock()
    mock_loop.run_until_complete.return_value = {"id": "test_collection_id"}
    mock_get_loop.return_value = mock_loop

    result = runner.invoke(
        scrape,
        [
            "--library",
            "test_lib",
            "--url",
            "http://test.com",
            "--config",
            config_file,
        ],
    )
    assert result.exit_code == 0


@patch("asyncio.get_event_loop")
def test_list_collections_command(mock_get_loop, runner, config_file):
    # Настраиваем мок для event loop
    mock_loop = MagicMock()
    mock_loop.run_until_complete.return_value = [
        {"id": "test_collection_id", "name": "Test Collection"}
    ]
    mock_get_loop.return_value = mock_loop

    result = runner.invoke(
        list_collections, ["--format", "json", "--config", config_file]
    )
    assert result.exit_code == 0


@patch("asyncio.get_event_loop")
def test_create_collection_command(mock_get_loop, runner, config_file):
    # Настраиваем мок для event loop
    mock_loop = MagicMock()
    mock_loop.run_until_complete.return_value = {"id": "test_collection_id"}
    mock_get_loop.return_value = mock_loop

    result = runner.invoke(
        create_collection,
        [
            "--name",
            "test_collection",
            "--description",
            "Test description",
            "--config",
            config_file,
        ],
    )
    assert result.exit_code == 0


@patch("asyncio.get_event_loop")
def test_delete_collection_command(mock_get_loop, runner, config_file):
    # Настраиваем мок для event loop
    mock_loop = MagicMock()
    mock_loop.run_until_complete.return_value = {"success": True}
    mock_get_loop.return_value = mock_loop

    result = runner.invoke(
        delete_collection,
        ["--id", "test_collection", "--force", "--config", config_file],
    )
    assert result.exit_code == 0


@patch("asyncio.get_event_loop")
def test_list_documents_command(mock_get_loop, runner, config_file):
    # Настраиваем мок для event loop
    mock_loop = MagicMock()
    mock_loop.run_until_complete.return_value = [
        {"id": "test_doc_id", "title": "Test Document"}
    ]
    mock_get_loop.return_value = mock_loop

    result = runner.invoke(
        list_documents,
        [
            "--collection",
            "test_collection",
            "--format",
            "json",
            "--config",
            config_file,
        ],
    )
    assert result.exit_code == 0


@patch("asyncio.get_event_loop")
def test_get_document_command(mock_get_loop, runner, config_file):
    # Настраиваем мок для event loop
    mock_loop = MagicMock()
    mock_loop.run_until_complete.return_value = {
        "id": "test_doc_id",
        "content": "Test content",
    }
    mock_get_loop.return_value = mock_loop

    result = runner.invoke(
        get_document,
        ["--id", "test_doc", "--format", "json", "--config", config_file],
    )
    assert result.exit_code == 0


@patch("asyncio.get_event_loop")
def test_delete_document_command(mock_get_loop, runner, config_file):
    # Настраиваем мок для event loop
    mock_loop = MagicMock()
    mock_loop.run_until_complete.return_value = {"success": True}
    mock_get_loop.return_value = mock_loop

    result = runner.invoke(
        delete_document,
        ["--id", "test_doc", "--force", "--config", config_file],
    )
    assert result.exit_code == 0


@patch("asyncio.get_event_loop")
def test_generate_report_command(mock_get_loop, runner, config_file):
    # Настраиваем мок для event loop
    mock_loop = MagicMock()
    # Мокаем результаты для методов _get_collections и _get_documents
    mock_loop.run_until_complete.side_effect = [
        # Мок для _get_collections - список словарей
        [{"id": "test-collection", "name": "Test Collection"}],
        # Мок для _get_documents - список словарей
        [
            {
                "id": "test-doc",
                "title": "Test Document",
                "collection_id": "test-collection",
            }
        ],
    ]
    mock_get_loop.return_value = mock_loop

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        result = runner.invoke(
            generate_report,
            ["--format", "html", "--output", f.name, "--config", config_file],
        )
        assert result.exit_code == 0
        assert os.path.exists(f.name)
        os.unlink(f.name)
