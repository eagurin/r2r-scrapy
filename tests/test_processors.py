import pytest
from r2r_scrapy.processors.code_processor import CodeProcessor
from r2r_scrapy.processors.markdown_processor import MarkdownProcessor
from r2r_scrapy.processors.html_processor import HTMLProcessor
from r2r_scrapy.processors.api_processor import APIDocProcessor

def test_code_processor():
    processor = CodeProcessor()
    
    # Test Python code
    python_code = """
def test_function():
    print("Hello")
    return True
"""
    result = processor.process_code(python_code, 'python')
    assert 'original' in result
    assert 'highlighted' in result
    assert result['language'] == 'python'
    
    # Test language detection
    js_code = """
function testFunction() {
    console.log("Hello");
    return true;
}
"""
    result = processor.process_code(js_code)
    assert result['language'] in ['javascript', 'js']
    
    # Test code cleaning
    indented_code = """
    def test():
        print("Hello")
        return True
    """
    cleaned = processor.clean_code(indented_code)
    assert not cleaned.startswith(' ')

def test_markdown_processor():
    processor = MarkdownProcessor()
    
    # Test basic markdown
    markdown = """
# Title
## Subtitle
Some text with **bold** and *italic*.
"""
    content, metadata = processor.process_markdown(markdown)
    assert 'Title' in content
    assert 'bold' in content
    assert 'italic' in content
    assert metadata.get('title') == 'Title'
    
    # Test code blocks
    markdown_with_code = """
# Test
```python
def test():
    return True
```
"""
    content, metadata = processor.process_markdown(markdown_with_code)
    assert 'python' in content
    assert 'def test()' in content

def test_html_processor():
    processor = HTMLProcessor()
    
    # Test HTML cleaning
    html = """
<html>
<head><title>Test</title></head>
<body>
    <nav>Navigation</nav>
    <div class="content">
        <h1>Title</h1>
        <p>Text</p>
    </div>
    <footer>Footer</footer>
</body>
</html>
"""
    content, metadata = processor.process(None, html)
    assert 'Navigation' not in content
    assert 'Footer' not in content
    assert 'Title' in content
    assert 'Text' in content
    
    # Test metadata extraction
    html_with_meta = """
<html>
<head>
    <title>Test Page</title>
    <meta name="description" content="Test description">
    <meta name="keywords" content="test,keywords">
</head>
<body>
    <div class="content">Content</div>
</body>
</html>
"""
    content, metadata = processor.process(None, html_with_meta)
    assert metadata.get('title') == 'Test Page'
    assert metadata.get('description') == 'Test description'
    assert 'test' in metadata.get('keywords', [])

def test_api_processor():
    processor = APIDocProcessor()
    
    # Test API element extraction
    api_doc = """
<div class="api-doc">
    <h2>function test_api(param1, param2)</h2>
    <p>Test API function</p>
    <pre><code>
def test_api(param1, param2):
    return param1 + param2
    </code></pre>
</div>
"""
    elements = processor.extract_api_elements(api_doc)
    assert len(elements) > 0
    assert elements[0]['type'] == 'function'
    assert elements[0]['name'] == 'test_api'
    assert 'param1' in elements[0]['params']
    
    # Test structure detection
    structure = processor.detect_structure({'css': lambda x: api_doc if x == '.api-documentation' else None})
    assert structure['main_content'] == api_doc 