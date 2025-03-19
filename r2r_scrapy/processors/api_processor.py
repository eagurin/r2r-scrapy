import re
from bs4 import BeautifulSoup
from r2r_scrapy.processors.code_processor import CodeProcessor

class APIDocProcessor:
    """Process API documentation"""
    
    def __init__(self):
        self.code_processor = CodeProcessor()
    
    def detect_structure(self, response):
        """Detect the structure of API documentation"""
        # Try to identify main content area
        content_selectors = [
            'main', 'article', '.content', '.documentation',
            '.api-documentation', '.api-reference', '#content',
        ]
        
        for selector in content_selectors:
            main_content = response.css(f"{selector}").get()
            if main_content:
                return {
                    'main_content': main_content,
                    'selector': selector
                }
        
        # Fallback to body if no specific content area found
        return {
            'main_content': response.css('body').get(),
            'selector': 'body'
        }
    
    def extract_api_elements(self, html_content):
        """Extract API elements (functions, methods, classes) from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        api_elements = []
        
        # Extract function/method definitions
        function_patterns = [
            # Common function definition patterns
            (r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)', 'function'),  # Python
            (r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)', 'function'),  # JavaScript
            (r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*function\s*\((.*?)\)', 'function'),  # JavaScript
            (r'public\s+(?:static\s+)?[a-zA-Z_][a-zA-Z0-9_]*\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)', 'method'),  # Java
        ]
        
        # Extract class definitions
        class_patterns = [
            (r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', 'class'),  # Python
            (r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:extends|implements)', 'class'),  # JavaScript/Java
            (r'interface\s+([a-zA-Z_][a-zA-Z0-9_]*)', 'interface'),  # Java/TypeScript
        ]
        
        # Find code blocks
        code_blocks = soup.find_all(['pre', 'code'])
        for block in code_blocks:
            code_text = block.get_text()
            
            # Process code with code processor
            language = block.get('class', [''])[0].replace('language-', '') if block.get('class') else None
            processed_code = self.code_processor.process_code(code_text, language)
            
            # Extract API elements from code
            for pattern, element_type in function_patterns + class_patterns:
                for match in re.finditer(pattern, code_text):
                    name = match.group(1)
                    params = match.group(2) if element_type == 'function' or element_type == 'method' else None
                    
                    api_elements.append({
                        'type': element_type,
                        'name': name,
                        'params': params,
                        'code': processed_code,
                    })
        
        # Find API elements in headings and descriptions
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        for heading in headings:
            heading_text = heading.get_text().strip()
            
            # Check if heading contains API element
            for pattern, element_type in function_patterns + class_patterns:
                match = re.search(pattern, heading_text)
                if match:
                    name = match.group(1)
                    params = match.group(2) if element_type == 'function' or element_type == 'method' else None
                    
                    # Get description (next sibling paragraphs)
                    description = ""
                    next_element = heading.next_sibling
                    while next_element and next_element.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                        if next_element.name == 'p':
                            description += next_element.get_text().strip() + "\n\n"
                        next_element = next_element.next_sibling
                    
                    api_elements.append({
                        'type': element_type,
                        'name': name,
                        'params': params,
                        'description': description.strip(),
                    })
        
        return api_elements
    
    def process(self, response):
        """Process API documentation"""
        # Detect structure
        structure = self.detect_structure(response)
        
        # Extract main content
        main_content = structure['main_content']
        
        # Extract API elements
        api_elements = self.extract_api_elements(main_content)
        
        # Clean HTML content
        soup = BeautifulSoup(main_content, 'html.parser')
        
        # Remove navigation, sidebars, footers
        for element in soup.select('nav, .sidebar, .navigation, footer, .footer, .menu'):
            element.decompose()
        
        # Extract metadata
        metadata = {
            'title': response.css('title::text').get(),
            'api_elements': [elem['name'] for elem in api_elements],
            'api_element_count': len(api_elements),
            'doc_type': 'api_reference',
        }
        
        # Convert to clean text
        content = soup.get_text(separator='\n\n')
        
        # Clean up whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r'\s+', ' ', content)
        
        return content, metadata 