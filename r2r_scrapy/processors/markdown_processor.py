import re
import markdown
from bs4 import BeautifulSoup
import html2text

class MarkdownProcessor:
    """Process Markdown content"""
    
    def __init__(self):
        self.html2text = html2text.HTML2Text()
        self.html2text.ignore_links = False
        self.html2text.ignore_images = False
        self.html2text.ignore_tables = False
        self.html2text.body_width = 0  # No wrapping
    
    def process_markdown(self, content):
        """Process Markdown content and extract metadata"""
        if not content:
            return content, {}
        
        # Extract metadata from frontmatter
        metadata = self._extract_metadata(content)
        
        # Clean up markdown
        cleaned_content = self.clean_markdown(content)
        
        # Convert to HTML for processing
        html = markdown.markdown(cleaned_content, extensions=['extra', 'meta', 'toc'])
        
        # Extract additional metadata from HTML
        metadata.update(self._extract_html_metadata(html))
        
        # Clean up HTML and convert back to Markdown
        cleaned_html = self._clean_html(html)
        final_markdown = self.html2text.handle(cleaned_html)
        
        return final_markdown, metadata
    
    def clean_markdown(self, content):
        """Clean up Markdown content"""
        # Remove frontmatter
        content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
        
        # Remove excessive newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Fix code block formatting
        content = re.sub(r'```\s+', '```\n', content)
        content = re.sub(r'\s+```', '\n```', content)
        
        # Fix list formatting
        content = re.sub(r'(\n[*-]\s+[^\n]+)(\n[^\n*-])', r'\1\n\2', content)
        
        return content
    
    def _extract_metadata(self, content):
        """Extract metadata from Markdown frontmatter"""
        metadata = {}
        
        # Look for frontmatter
        frontmatter_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            
            # Parse key-value pairs
            for line in frontmatter.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Handle lists
                    if value.startswith('[') and value.endswith(']'):
                        value = [v.strip() for v in value[1:-1].split(',')]
                    
                    metadata[key] = value
        
        return metadata
    
    def _extract_html_metadata(self, html):
        """Extract metadata from HTML content"""
        metadata = {}
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title
        title = soup.find('h1')
        if title:
            metadata['title'] = title.get_text()
        
        # Extract headings
        headings = []
        for level in range(1, 7):
            for heading in soup.find_all(f'h{level}'):
                headings.append({
                    'level': level,
                    'text': heading.get_text().strip()
                })
        metadata['headings'] = headings
        
        # Extract links
        links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href:
                links.append({
                    'text': link.get_text(),
                    'url': href
                })
        metadata['links'] = links
        
        return metadata
    
    def _clean_html(self, html):
        """Clean up HTML content"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for element in soup.select('script, style'):
            element.decompose()
        
        # Clean up whitespace
        for element in soup.find_all(text=True):
            if element.parent.name not in ['pre', 'code']:
                element.replace_with(element.string.strip())
        
        return str(soup) 