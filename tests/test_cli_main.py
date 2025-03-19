import pytest
from click.testing import CliRunner
from r2r_scrapy.cli.main import cli

@pytest.fixture
def runner():
    return CliRunner()

def test_cli_help(runner):
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'R2R Scrapy - Asynchronous Documentation Collector' in result.output

def test_cli_version(runner):
    result = runner.invoke(cli, ['--version'])
    assert result.exit_code == 0
    assert '1.0.0' in result.output

def test_cli_commands_available(runner):
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    
    # Check if all commands are listed
    commands = [
        'scrape',
        'list-collections',
        'create-collection',
        'delete-collection',
        'list-documents',
        'get-document',
        'delete-document',
        'generate-report'
    ]
    
    for command in commands:
        assert command in result.output

def test_cli_command_help(runner):
    commands = [
        'scrape',
        'list-collections',
        'create-collection',
        'delete-collection',
        'list-documents',
        'get-document',
        'delete-document',
        'generate-report'
    ]
    
    for command in commands:
        result = runner.invoke(cli, [command, '--help'])
        assert result.exit_code == 0
        assert 'Usage:' in result.output
        assert 'Options:' in result.output

def test_cli_invalid_command(runner):
    result = runner.invoke(cli, ['invalid-command'])
    assert result.exit_code == 2
    assert 'No such command' in result.output

def test_cli_missing_required_args(runner):
    # Test scrape command without required args
    result = runner.invoke(cli, ['scrape'])
    assert result.exit_code == 2
    assert 'Missing option' in result.output
    
    # Test create-collection command without required args
    result = runner.invoke(cli, ['create-collection'])
    assert result.exit_code == 2
    assert 'Missing option' in result.output
    
    # Test delete-collection command without required args
    result = runner.invoke(cli, ['delete-collection'])
    assert result.exit_code == 2
    assert 'Missing option' in result.output
    
    # Test get-document command without required args
    result = runner.invoke(cli, ['get-document'])
    assert result.exit_code == 2
    assert 'Missing option' in result.output
    
    # Test delete-document command without required args
    result = runner.invoke(cli, ['delete-document'])
    assert result.exit_code == 2
    assert 'Missing option' in result.output 