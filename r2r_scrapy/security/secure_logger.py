import logging
import re
import json
import os
from datetime import datetime

class SecureLogger:
    """Secure logging with sensitive data masking"""
    
    def __init__(self, log_dir=None, log_level=logging.INFO, mask_sensitive=True):
        self.log_dir = log_dir or os.path.join(os.getcwd(), 'logs')
        self.mask_sensitive = mask_sensitive
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger('r2r_scrapy')
        self.logger.setLevel(log_level)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create file handler
        log_file = os.path.join(self.log_dir, f"r2r_scrapy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        # Set up sensitive data patterns
        self.sensitive_patterns = [
            (r'api_key\s*[=:]\s*["\']([^"\']+)["\']', 'api_key=***'),
            (r'password\s*[=:]\s*["\']([^"\']+)["\']', 'password=***'),
            (r'token\s*[=:]\s*["\']([^"\']+)["\']', 'token=***'),
            (r'secret\s*[=:]\s*["\']([^"\']+)["\']', 'secret=***'),
            (r'auth\s*[=:]\s*["\']([^"\']+)["\']', 'auth=***'),
            (r'bearer\s+([a-zA-Z0-9._-]+)', 'bearer ***'),
        ]
    
    def debug(self, message, *args, **kwargs):
        """Log debug message with sensitive data masking"""
        if self.mask_sensitive:
            message = self._mask_sensitive_data(message)
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message, *args, **kwargs):
        """Log info message with sensitive data masking"""
        if self.mask_sensitive:
            message = self._mask_sensitive_data(message)
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message, *args, **kwargs):
        """Log warning message with sensitive data masking"""
        if self.mask_sensitive:
            message = self._mask_sensitive_data(message)
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message, *args, **kwargs):
        """Log error message with sensitive data masking"""
        if self.mask_sensitive:
            message = self._mask_sensitive_data(message)
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message, *args, **kwargs):
        """Log critical message with sensitive data masking"""
        if self.mask_sensitive:
            message = self._mask_sensitive_data(message)
        self.logger.critical(message, *args, **kwargs)
    
    def _mask_sensitive_data(self, message):
        """Mask sensitive data in log message"""
        if isinstance(message, dict) or isinstance(message, list):
            # Convert to JSON string for masking
            message = json.dumps(message)
            masked_message = self._mask_json(message)
            return masked_message
        
        if not isinstance(message, str):
            # Convert to string
            message = str(message)
        
        # Apply masking patterns
        masked_message = message
        for pattern, replacement in self.sensitive_patterns:
            masked_message = re.sub(pattern, replacement, masked_message, flags=re.IGNORECASE)
        
        return masked_message
    
    def _mask_json(self, json_str):
        """Mask sensitive data in JSON string"""
        try:
            # Parse JSON
            data = json.loads(json_str)
            
            # Mask sensitive fields
            if isinstance(data, dict):
                self._mask_dict(data)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        self._mask_dict(item)
            
            # Convert back to JSON
            return json.dumps(data)
        except Exception:
            # If not valid JSON, treat as string
            return json_str
    
    def _mask_dict(self, data):
        """Recursively mask sensitive fields in dictionary"""
        sensitive_keys = ['api_key', 'password', 'token', 'secret', 'auth', 'authorization']
        
        for key in list(data.keys()):
            # Check if key is sensitive
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                data[key] = '***'
            elif isinstance(data[key], dict):
                self._mask_dict(data[key])
            elif isinstance(data[key], list):
                for item in data[key]:
                    if isinstance(item, dict):
                        self._mask_dict(item) 