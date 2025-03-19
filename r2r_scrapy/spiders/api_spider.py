import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from r2r_scrapy.processors import APIDocProcessor

class APIDocSpider(CrawlSpider):
    name = 'api_doc_spider'
    
    def __init__(self, domain=None, start_urls=None, allowed_paths=None, *args, **kwargs):
        super(APIDocSpider, self).__init__(*args, **kwargs)
        self.allowed_domains = [domain] if domain else []
        self.start_urls = start_urls.split(',') if start_urls else []
        
        # Rules for following only API documentation links
        self.rules = (
            Rule(
                LinkExtractor(
                    allow=allowed_paths.split(',') if allowed_paths else (),
                    deny=('search', 'print', 'pdf', 'zip', 'download')
                ),
                callback='parse_api_doc',
                follow=True
            ),
        )
        
        self.processor = APIDocProcessor()
    
    def parse_api_doc(self, response):
        # Detect structure of documentation
        structure = self.processor.detect_structure(response)
        
        # Extract main content
        main_content = structure['main_content']
        
        # Extract API elements (functions, methods, classes)
        api_elements = self.processor.extract_api_elements(main_content)
        
        # Create item with processed content
        yield {
            'url': response.url,
            'title': response.css('title::text').get(),
            'content': main_content,
            'api_elements': api_elements,
            'metadata': {
                'library_name': self.settings.get('LIBRARY_NAME'),
                'version': self.extract_version(response),
                'doc_type': 'api_reference',
                'language': self.detect_programming_language(response),
            }
        }
    
    def extract_version(self, response):
        # Try to extract version from common patterns
        version = response.css('.version::text, .doc-version::text').get()
        if not version:
            # Try regex patterns for version extraction
            import re
            text = ' '.join(response.css('body ::text').getall())
            match = re.search(r'version\s+(\d+\.\d+\.\d+)', text, re.I)
            if match:
                version = match.group(1)
        return version or 'unknown'
    
    def detect_programming_language(self, response):
        # Detect programming language based on content
        code_blocks = response.css('pre code::text').getall()
        if not code_blocks:
            return 'unknown'
        
        # Simple heuristic for language detection
        text = ' '.join(code_blocks)
        if 'def ' in text or 'import ' in text:
            return 'python'
        elif 'function ' in text or 'var ' in text or 'const ' in text:
            return 'javascript'
        elif 'public class ' in text or 'private ' in text:
            return 'java'
        return 'unknown' 