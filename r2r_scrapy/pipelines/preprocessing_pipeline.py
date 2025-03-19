import logging
from bs4 import BeautifulSoup
import re
import html2text

class PreprocessingPipeline:
    """Pipeline for preprocessing scraped content"""
    
    def __init__(self, settings):
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        
        # Initialize HTML to text converter
        self.html2text = html2text.HTML2Text()
        self.html2text.ignore_links = False
        self.html2text.ignore_images = False
        self.html2text.ignore_tables = False
        self.html2text.body_width = 0  # No wrapping
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)
    
    def process_item(self, item, spider):
        """Process a scraped item"""
        # Skip if already processed
        if item.get('preprocessed'):
            return item
        
        # Extract content
        content = item.get('content', '')
        
        # Skip empty content
        if not content:
            self.logger.warning(f"Empty content for URL: {item.get('url')}")
            return item
        
        # Detect content type
        content_type = self._detect_content_type(content, item)
        
        # Process based on content type
        if content_type == 'html':
            processed_content = self._process_html(content)
        elif content_type == 'markdown':
            processed_content = self._process_markdown(content)
        elif content_type == 'text':
            processed_content = self._process_text(content)
        else:
            processed_content = content
        
        # Update item
        item['content'] = processed_content
        item['content_type'] = content_type
        item['preprocessed'] = True
        
        # Extract metadata if not already present
        if not item.get('metadata'):
            item['metadata'] = self._extract_metadata(content, content_type, item)
        
        return item
    
    def _detect_content_type(self, content, item):
        """Detect content type (HTML, Markdown, text)"""
        # Check URL for clues
        url = item.get('url', '')
        if url.endswith('.md') or url.endswith('.markdown'):
            return 'markdown'
        elif url.endswith('.txt'):
            return 'text'
        
        # Check for HTML tags
        if re.search(r'<(?:html|body|div|p|h[1-6]|ul|ol|table)\b', content):
            return 'html'
        
        # Check for Markdown indicators
        markdown_indicators = ['##', '```', '**', '==', '- [ ]', '[toc]']
        if any(indicator in content for indicator in markdown_indicators):
            return 'markdown'
        
        # Default to text
        return 'text'
    
    def _process_html(self, content):
        """Process HTML content"""
        # Parse HTML
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove unwanted elements
        for selector in ['script', 'style', 'nav', 'footer', '.navigation', '.menu', '.ads', '.sidebar']:
            for element in soup.select(selector):
                element.decompose()
        
        # Convert to Markdown
        markdown = self.html2text.handle(str(soup))
        
        # Clean up Markdown
        cleaned = self._clean_markdown(markdown)
        
        return cleaned
    
    def _process_markdown(self, content):
        """Process Markdown content"""
        # Clean up Markdown
        cleaned = self._clean_markdown(content)
        
        return cleaned
    
    def _process_text(self, content):
        """Process plain text content"""
        # Clean up text
        cleaned = content.strip()
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        
        return cleaned
    
    def _clean_markdown(self, markdown):
        """Clean up Markdown content"""
        # Remove excessive newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', markdown)
        
        # Fix code block formatting
        cleaned = re.sub(r'```\s+', '```\n', cleaned)
        cleaned = re.sub(r'\s+```', '\n```', cleaned)
        
        # Fix list formatting
        cleaned = re.sub(r'(\n[*-]\s+[^\n]+)(\n[^\n*-])', r'\1\n\2', cleaned)
        
        return cleaned
    
    def _extract_metadata(self, content, content_type, item):
        """Extract metadata from content"""
        metadata = {}
        
        # Add basic metadata
        metadata['url'] = item.get('url', '')
        metadata['title'] = item.get('title', '')
        metadata['content_type'] = content_type
        
        # Extract more metadata based on content type
        if content_type == 'html':
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract description
            description_meta = soup.find('meta', attrs={'name': 'description'})
            if description_meta:
                metadata['description'] = description_meta.get('content', '')
            
            # Extract keywords
            keywords_meta = soup.find('meta', attrs={'name': 'keywords'})
            if keywords_meta:
                metadata['keywords'] = [k.strip() for k in keywords_meta.get('content', '').split(',')]
            
            # Extract headings
            headings = []
            for level in range(1, 7):
                for heading in soup.find_all(f'h{level}'):
                    headings.append({
                        'level': level,
                        'text': heading.get_text().strip()
                    })
            
            metadata['headings'] = headings
        
        # Add library name from spider settings
        metadata['library_name'] = self.settings.get('LIBRARY_NAME', '')
        
        return metadata 