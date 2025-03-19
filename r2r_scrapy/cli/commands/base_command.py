import click
import logging
import os
from r2r_scrapy.config import Config

class BaseCommand:
    """Base class for CLI commands"""
    
    def __init__(self, config_path=None):
        # Load configuration
        self.config = Config(config_path or os.environ.get('R2R_SCRAPY_CONFIG', 'config.yaml'))
        
        # Set up logging
        self.logger = logging.getLogger('r2r_scrapy')
    
    def get_r2r_api_settings(self):
        """Get R2R API settings from config"""
        r2r_api_url = self.config.get('r2r.api_url') or os.environ.get('R2R_API_URL')
        r2r_api_key = self.config.get('r2r.api_key') or os.environ.get('R2R_API_KEY')
        
        if not r2r_api_url or not r2r_api_key:
            raise ValueError("R2R API URL and API key must be provided in config or environment variables")
        
        return {
            'api_url': r2r_api_url,
            'api_key': r2r_api_key,
            'batch_size': self.config.get('r2r.batch_size', 10),
            'max_concurrency': self.config.get('r2r.max_concurrency', 5),
        }
    
    def get_scrapy_settings(self):
        """Get Scrapy settings from config"""
        return {
            'concurrent_requests': self.config.get('scrapy.concurrent_requests', 16),
            'concurrent_requests_per_domain': self.config.get('scrapy.concurrent_requests_per_domain', 8),
            'download_delay': self.config.get('scrapy.download_delay', 0.5),
            'user_agent': self.config.get('scrapy.user_agent', 'R2R Scrapy/1.0 (+https://github.com/eagurin/r2r-scrapy)'),
            'javascript_rendering': self.config.get('scrapy.javascript_rendering', False),
            'splash_url': self.config.get('scrapy.splash_url', 'http://localhost:8050'),
        }
    
    def get_processing_settings(self):
        """Get processing settings from config"""
        return {
            'default_chunking_strategy': self.config.get('processing.default_chunking_strategy', 'semantic'),
            'chunk_size': self.config.get('processing.chunk_size', 800),
            'chunk_overlap': self.config.get('processing.chunk_overlap', 150),
            'preserve_code_blocks': self.config.get('processing.preserve_code_blocks', True),
            'extract_metadata': self.config.get('processing.extract_metadata', True),
        }
    
    def get_monitoring_settings(self):
        """Get monitoring settings from config"""
        return {
            'enabled': self.config.get('monitoring.enabled', True),
            'prometheus_port': self.config.get('monitoring.prometheus_port', 9090),
            'quality_threshold': self.config.get('monitoring.quality_threshold', 0.8),
            'alert_on_error': self.config.get('monitoring.alert_on_error', True),
        } 