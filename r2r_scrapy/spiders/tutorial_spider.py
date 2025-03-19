import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from r2r_scrapy.processors import MarkdownProcessor, HTMLProcessor

class TutorialSpider(CrawlSpider):
    name = 'tutorial_spider'
    
    def __init__(self, domain=None, start_urls=None, allowed_paths=None, *args, **kwargs):
        super(TutorialSpider, self).__init__(*args, **kwargs)
        self.allowed_domains = [domain] if domain else []
        self.start_urls = start_urls.split(',') if start_urls else []
        
        # Rules for following tutorial links
        self.rules = (
            Rule(
                LinkExtractor(
                    allow=allowed_paths.split(',') if allowed_paths else (),
                    deny=('search', 'print', 'pdf', 'zip', 'download'),
                    restrict_css=('.tutorial', '.guide', '.docs', '.documentation')
                ),
                callback='parse_tutorial',
                follow=True
            ),
        )
        
        self.markdown_processor = MarkdownProcessor()
        self.html_processor = HTMLProcessor()
    
    def parse_tutorial(self, response):
        # Detect content type (HTML or Markdown)
        content_type = self.detect_content_type(response)
        
        # Process content based on type
        if content_type == 'markdown':
            content, metadata = self.markdown_processor.process(response)
        else:  # HTML
            content, metadata = self.html_processor.process(response)
        
        # Extract tutorial structure (headings, sections)
        structure = self.extract_structure(content)
        
        # Create item with processed content
        yield {
            'url': response.url,
            'title': metadata.get('title') or response.css('title::text').get(),
            'content': content,
            'structure': structure,
            'metadata': {
                **metadata,
                'library_name': self.settings.get('LIBRARY_NAME'),
                'doc_type': 'tutorial',
                'level': self.detect_tutorial_level(content),
                'estimated_reading_time': self.calculate_reading_time(content),
            }
        }
    
    def detect_content_type(self, response):
        """Detect if content is Markdown or HTML"""
        # Check content type header
        content_type = response.headers.get('Content-Type', b'').decode('utf-8', 'ignore')
        if 'text/markdown' in content_type or 'application/markdown' in content_type:
            return 'markdown'
        
        # Check URL extension
        if response.url.endswith('.md'):
            return 'markdown'
        
        # Check for Markdown indicators in content
        text = ' '.join(response.css('body ::text').getall())
        markdown_indicators = ['##', '```', '**', '==', '- [ ]', '[toc]']
        if any(indicator in text for indicator in markdown_indicators):
            return 'markdown'
        
        return 'html'
    
    def extract_structure(self, content):
        """Extract structure (headings, sections) from content"""
        import re
        
        # Extract headings
        headings = []
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+?)(?:\s+#{1,6})?$', re.MULTILINE)
        for match in heading_pattern.finditer(content):
            level = len(match.group(1))
            text = match.group(2).strip()
            headings.append({
                'level': level,
                'text': text,
                'position': match.start()
            })
        
        # Build hierarchical structure
        structure = []
        stack = [{'level': 0, 'children': structure}]
        
        for heading in headings:
            while heading['level'] <= stack[-1]['level']:
                stack.pop()
            
            current = {
                'title': heading['text'],
                'level': heading['level'],
                'position': heading['position'],
                'children': []
            }
            
            stack[-1]['children'].append(current)
            stack.append(current)
        
        return structure
    
    def detect_tutorial_level(self, content):
        """Detect tutorial difficulty level"""
        # Simple heuristic based on keywords
        beginner_keywords = ['introduction', 'getting started', 'basics', 'beginner', 'first steps']
        intermediate_keywords = ['intermediate', 'advanced topics', 'deeper', 'in-depth']
        advanced_keywords = ['advanced', 'expert', 'deep dive', 'internals', 'optimization']
        
        content_lower = content.lower()
        
        # Count occurrences of level-indicating keywords
        beginner_count = sum(content_lower.count(kw) for kw in beginner_keywords)
        intermediate_count = sum(content_lower.count(kw) for kw in intermediate_keywords)
        advanced_count = sum(content_lower.count(kw) for kw in advanced_keywords)
        
        # Determine level based on keyword counts
        if advanced_count > intermediate_count and advanced_count > beginner_count:
            return 'advanced'
        elif intermediate_count > beginner_count:
            return 'intermediate'
        else:
            return 'beginner'
    
    def calculate_reading_time(self, content):
        """Calculate estimated reading time in minutes"""
        # Average reading speed: 200-250 words per minute
        words = len(content.split())
        reading_time = max(1, round(words / 225))
        return reading_time 