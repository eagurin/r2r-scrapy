import re
from urllib.parse import urlparse
import logging

class URLPrioritizer:
    """Prioritize URLs for crawling based on relevance to documentation"""
    
    def __init__(self, settings=None):
        self.logger = logging.getLogger(__name__)
        
        # Default settings
        self.settings = settings or {}
        
        # Priority patterns
        self.high_priority_patterns = self.settings.get('HIGH_PRIORITY_PATTERNS', [
            r'/docs?/',
            r'/api/',
            r'/reference/',
            r'/guide/',
            r'/tutorial/',
            r'/examples?/',
            r'/getting-started/',
        ])
        
        self.medium_priority_patterns = self.settings.get('MEDIUM_PRIORITY_PATTERNS', [
            r'/blog/',
            r'/articles?/',
            r'/faq/',
            r'/help/',
            r'/support/',
        ])
        
        self.low_priority_patterns = self.settings.get('LOW_PRIORITY_PATTERNS', [
            r'/about/',
            r'/contact/',
            r'/team/',
            r'/pricing/',
            r'/download/',
        ])
        
        self.ignore_patterns = self.settings.get('IGNORE_PATTERNS', [
            r'/search/',
            r'/login/',
            r'/signup/',
            r'/register/',
            r'/account/',
            r'/cart/',
            r'/checkout/',
        ])
        
        # File extension priorities
        self.high_priority_extensions = self.settings.get('HIGH_PRIORITY_EXTENSIONS', [
            '.md', '.rst', '.txt', '.html', '.htm'
        ])
        
        self.medium_priority_extensions = self.settings.get('MEDIUM_PRIORITY_EXTENSIONS', [
            '.py', '.js', '.java', '.c', '.cpp', '.cs', '.go', '.rb', '.php'
        ])
        
        self.low_priority_extensions = self.settings.get('LOW_PRIORITY_EXTENSIONS', [
            '.json', '.yaml', '.yml', '.xml', '.csv'
        ])
        
        self.ignore_extensions = self.settings.get('IGNORE_EXTENSIONS', [
            '.pdf', '.zip', '.tar', '.gz', '.rar', '.exe', '.bin',
            '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico',
            '.mp3', '.mp4', '.avi', '.mov', '.wmv',
        ])
    
    def get_priority(self, url):
        """Get priority score for a URL (higher is more important)"""
        # Check if URL should be ignored
        if self._matches_patterns(url, self.ignore_patterns) or self._has_extension(url, self.ignore_extensions):
            return 0
        
        # Start with base priority
        priority = 100
        
        # Adjust based on path patterns
        if self._matches_patterns(url, self.high_priority_patterns):
            priority += 200
        elif self._matches_patterns(url, self.medium_priority_patterns):
            priority += 100
        elif self._matches_patterns(url, self.low_priority_patterns):
            priority += 50
        
        # Adjust based on file extension
        if self._has_extension(url, self.high_priority_extensions):
            priority += 100
        elif self._has_extension(url, self.medium_priority_extensions):
            priority += 50
        elif self._has_extension(url, self.low_priority_extensions):
            priority += 25
        
        # Adjust based on URL depth (prefer shallower URLs)
        depth = self._get_url_depth(url)
        priority -= depth * 10
        
        # Ensure priority is positive
        return max(1, priority)
    
    def prioritize_urls(self, urls):
        """Sort URLs by priority (highest first)"""
        return sorted(urls, key=self.get_priority, reverse=True)
    
    def _matches_patterns(self, url, patterns):
        """Check if URL matches any of the patterns"""
        for pattern in patterns:
            if re.search(pattern, url, re.I):
                return True
        return False
    
    def _has_extension(self, url, extensions):
        """Check if URL has one of the specified extensions"""
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        return any(path.endswith(ext) for ext in extensions)
    
    def _get_url_depth(self, url):
        """Get the depth of a URL (number of path segments)"""
        parsed_url = urlparse(url)
        path = parsed_url.path
        # Count path segments, ignoring empty segments
        segments = [s for s in path.split('/') if s]
        return len(segments)
