import logging
import re
import time
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server

class QualityMonitor:
    """Monitor quality of scraped content"""
    
    def __init__(self, port=9090):
        # Set up logging
        self.logger = logging.getLogger('r2r_scrapy_quality')
        self.logger.setLevel(logging.INFO)
        
        # Prometheus metrics
        self.pages_scraped = Counter('pages_scraped_total', 'Total number of pages scraped', ['status', 'doc_type'])
        self.content_size = Histogram('content_size_bytes', 'Size of scraped content in bytes', ['doc_type'])
        self.processing_time = Histogram('processing_time_seconds', 'Time spent processing a document', ['stage'])
        self.chunk_count = Histogram('chunk_count', 'Number of chunks per document', ['strategy'])
        self.r2r_response_time = Histogram('r2r_response_time_seconds', 'R2R API response time')
        self.r2r_errors = Counter('r2r_errors_total', 'Total number of R2R API errors', ['error_type'])
        
        # Start HTTP server for metrics
        start_http_server(port)
    
    def record_page_scraped(self, status, doc_type):
        """Record metric for scraped page"""
        self.pages_scraped.labels(status=status, doc_type=doc_type).inc()
    
    def record_content_size(self, size, doc_type):
        """Record metric for content size"""
        self.content_size.labels(doc_type=doc_type).observe(size)
    
    def time_processing(self, stage):
        """Decorator for measuring processing time"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                processing_time = time.time() - start_time
                self.processing_time.labels(stage=stage).observe(processing_time)
                return result
            return wrapper
        return decorator
    
    def record_chunk_stats(self, chunk_count, strategy):
        """Record metric for chunk count"""
        self.chunk_count.labels(strategy=strategy).observe(chunk_count)
    
    def record_r2r_response(self, response_time, status_code):
        """Record metric for R2R API response"""
        self.r2r_response_time.observe(response_time)
        if status_code >= 400:
            error_type = 'client_error' if status_code < 500 else 'server_error'
            self.r2r_errors.labels(error_type=error_type).inc()
    
    def validate_document_quality(self, document):
        """Check quality of processed document"""
        issues = []
        
        # Check for minimum content size
        if len(document['content']) < 100:
            issues.append("Content too short")
        
        # Check for required metadata
        required_metadata = ['title', 'url']
        for field in required_metadata:
            if field not in document['metadata'] or not document['metadata'][field]:
                issues.append(f"Missing required metadata: {field}")
        
        # Check for HTML tags in cleaned content
        if re.search(r'<[^>]+>', document['content']):
            issues.append("HTML tags found in cleaned content")
        
        # Log issues if any
        if issues:
            self.logger.warning(f"Document quality issues: {', '.join(issues)}")
            return False
        
        return True 