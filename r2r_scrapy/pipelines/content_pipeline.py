import logging
from r2r_scrapy.processors.code_processor import CodeProcessor
from r2r_scrapy.processors.markdown_processor import MarkdownProcessor
from r2r_scrapy.processors.html_processor import HTMLProcessor
from r2r_scrapy.processors.api_processor import APIDocProcessor

class ContentPipeline:
    """Pipeline for processing content with specialized processors"""
    
    def __init__(self, settings):
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        
        # Initialize processors
        self.code_processor = CodeProcessor()
        self.markdown_processor = MarkdownProcessor()
        self.html_processor = HTMLProcessor()
        self.api_processor = APIDocProcessor()
        
        # Settings
        self.extract_code_blocks = settings.getbool('EXTRACT_CODE_BLOCKS', True)
        self.process_api_elements = settings.getbool('PROCESS_API_ELEMENTS', True)
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)
    
    def process_item(self, item, spider):
        """Process a scraped item with specialized processors"""
        # Skip if not preprocessed
        if not item.get('preprocessed'):
            return item
        
        # Skip if already processed by content pipeline
        if item.get('content_processed'):
            return item
        
        # Get content and type
        content = item.get('content', '')
        content_type = item.get('content_type', 'text')
        
        # Skip empty content
        if not content:
            return item
        
        # Process based on content type
        if content_type == 'markdown':
            processed_content, metadata = self.markdown_processor.process_markdown(content)
        elif content_type == 'html':
            processed_content, metadata = self.html_processor.process(None, content)
        else:
            processed_content = content
            metadata = {}
        
        # Extract code blocks if enabled
        if self.extract_code_blocks:
            code_blocks = self._extract_code_blocks(processed_content)
            if code_blocks:
                metadata['code_blocks'] = code_blocks
        
        # Process API elements if enabled and doc_type is api_reference
        if self.process_api_elements and item.get('metadata', {}).get('doc_type') == 'api_reference':
            api_elements = self.api_processor.extract_api_elements(content)
            if api_elements:
                metadata['api_elements'] = api_elements
        
        # Update item
        item['content'] = processed_content
        item['content_processed'] = True
        
        # Update metadata
        item_metadata = item.get('metadata', {})
        item_metadata.update(metadata)
        item['metadata'] = item_metadata
        
        return item
    
    def _extract_code_blocks(self, content):
        """Extract and process code blocks from content"""
        import re
        
        # Match markdown code blocks
        code_blocks = []
        for match in re.finditer(r'```(\w*)\n([\s\S]*?)\n```', content):
            language = match.group(1) or None
            code = match.group(2)
            
            # Process code
            processed_code = self.code_processor.process_code(code, language)
            
            code_blocks.append({
                'language': language or 'text',
                'code': code,
                'processed': processed_code,
            })
        
        return code_blocks 