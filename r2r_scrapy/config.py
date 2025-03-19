import os
import yaml
from typing import Dict, Any, Optional

class Config:
    """Configuration management for R2R Scrapy"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = {}
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        
        # Override with environment variables
        self.load_from_env()
    
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Error loading configuration from file: {e}")
    
    def load_from_env(self) -> None:
        """Load configuration from environment variables"""
        # R2R Configuration
        if os.environ.get('R2R_API_KEY'):
            self.set_nested_value(['r2r', 'api_key'], os.environ.get('R2R_API_KEY'))
        
        if os.environ.get('R2R_API_URL'):
            self.set_nested_value(['r2r', 'api_url'], os.environ.get('R2R_API_URL'))
        
        if os.environ.get('R2R_BATCH_SIZE'):
            self.set_nested_value(['r2r', 'batch_size'], int(os.environ.get('R2R_BATCH_SIZE')))
        
        if os.environ.get('R2R_MAX_CONCURRENCY'):
            self.set_nested_value(['r2r', 'max_concurrency'], int(os.environ.get('R2R_MAX_CONCURRENCY')))
        
        # Scrapy Configuration
        if os.environ.get('SCRAPY_CONCURRENT_REQUESTS'):
            self.set_nested_value(['scrapy', 'concurrent_requests'], 
                                 int(os.environ.get('SCRAPY_CONCURRENT_REQUESTS')))
        
        if os.environ.get('SCRAPY_CONCURRENT_REQUESTS_PER_DOMAIN'):
            self.set_nested_value(['scrapy', 'concurrent_requests_per_domain'], 
                                 int(os.environ.get('SCRAPY_CONCURRENT_REQUESTS_PER_DOMAIN')))
        
        if os.environ.get('SCRAPY_DOWNLOAD_DELAY'):
            self.set_nested_value(['scrapy', 'download_delay'], 
                                 float(os.environ.get('SCRAPY_DOWNLOAD_DELAY')))
        
        if os.environ.get('SCRAPY_USER_AGENT'):
            self.set_nested_value(['scrapy', 'user_agent'], os.environ.get('SCRAPY_USER_AGENT'))
        
        # Processing Configuration
        if os.environ.get('PROCESSING_DEFAULT_CHUNKING_STRATEGY'):
            self.set_nested_value(['processing', 'default_chunking_strategy'], 
                                 os.environ.get('PROCESSING_DEFAULT_CHUNKING_STRATEGY'))
        
        if os.environ.get('PROCESSING_CHUNK_SIZE'):
            self.set_nested_value(['processing', 'chunk_size'], 
                                 int(os.environ.get('PROCESSING_CHUNK_SIZE')))
        
        if os.environ.get('PROCESSING_CHUNK_OVERLAP'):
            self.set_nested_value(['processing', 'chunk_overlap'], 
                                 int(os.environ.get('PROCESSING_CHUNK_OVERLAP')))
        
        if os.environ.get('PROCESSING_PRESERVE_CODE_BLOCKS'):
            self.set_nested_value(['processing', 'preserve_code_blocks'], 
                                 os.environ.get('PROCESSING_PRESERVE_CODE_BLOCKS').lower() == 'true')
        
        if os.environ.get('PROCESSING_EXTRACT_METADATA'):
            self.set_nested_value(['processing', 'extract_metadata'], 
                                 os.environ.get('PROCESSING_EXTRACT_METADATA').lower() == 'true')
        
        # Monitoring Configuration
        if os.environ.get('MONITORING_ENABLED'):
            self.set_nested_value(['monitoring', 'enabled'], 
                                 os.environ.get('MONITORING_ENABLED').lower() == 'true')
        
        if os.environ.get('MONITORING_PROMETHEUS_PORT'):
            self.set_nested_value(['monitoring', 'prometheus_port'], 
                                 int(os.environ.get('MONITORING_PROMETHEUS_PORT')))
        
        if os.environ.get('MONITORING_QUALITY_THRESHOLD'):
            self.set_nested_value(['monitoring', 'quality_threshold'], 
                                 float(os.environ.get('MONITORING_QUALITY_THRESHOLD')))
        
        if os.environ.get('MONITORING_ALERT_ON_ERROR'):
            self.set_nested_value(['monitoring', 'alert_on_error'], 
                                 os.environ.get('MONITORING_ALERT_ON_ERROR').lower() == 'true')
    
    def set_nested_value(self, keys: list, value: Any) -> None:
        """Set a nested value in the configuration dictionary"""
        current = self.config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the configuration"""
        keys = key.split('.')
        current = self.config
        
        for k in keys:
            if k not in current:
                return default
            current = current[k]
        
        return current
    
    def get_all(self) -> Dict[str, Any]:
        """Get the entire configuration"""
        return self.config
