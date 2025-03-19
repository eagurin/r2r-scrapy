import os
import pytest
import tempfile
import yaml
from r2r_scrapy.config import Config

def test_load_from_file():
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        config_data = {
            'r2r': {
                'api_url': 'http://test.com',
                'api_key': 'test_key',
                'batch_size': 10
            }
        }
        yaml.dump(config_data, f)
    
    config = Config(f.name)
    assert config.get('r2r.api_url') == 'http://test.com'
    assert config.get('r2r.api_key') == 'test_key'
    assert config.get('r2r.batch_size') == 10
    
    os.unlink(f.name)

def test_load_from_env():
    os.environ['R2R_API_KEY'] = 'env_key'
    os.environ['R2R_API_URL'] = 'http://env.com'
    os.environ['R2R_BATCH_SIZE'] = '20'
    
    config = Config()
    assert config.get('r2r.api_key') == 'env_key'
    assert config.get('r2r.api_url') == 'http://env.com'
    assert config.get('r2r.batch_size') == 20
    
    del os.environ['R2R_API_KEY']
    del os.environ['R2R_API_URL']
    del os.environ['R2R_BATCH_SIZE']

def test_get_default():
    config = Config()
    assert config.get('nonexistent.key', 'default') == 'default'

def test_get_nested():
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        config_data = {
            'nested': {
                'key1': {
                    'key2': 'value'
                }
            }
        }
        yaml.dump(config_data, f)
    
    config = Config(f.name)
    assert config.get('nested.key1.key2') == 'value'
    
    os.unlink(f.name)

def test_get_all():
    config_data = {
        'key1': 'value1',
        'key2': 'value2'
    }
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        yaml.dump(config_data, f)
    
    config = Config(f.name)
    assert config.get_all() == config_data
    
    os.unlink(f.name) 