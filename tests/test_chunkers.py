import pytest
from r2r_scrapy.chunkers.semantic_chunker import SemanticChunker
from r2r_scrapy.chunkers.code_chunker import CodeChunker
from r2r_scrapy.chunkers.markdown_chunker import MarkdownChunker
from r2r_scrapy.chunkers.recursive_chunker import RecursiveChunker

def test_semantic_chunker():
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)
    
    # Test basic text chunking
    text = "This is a test. " * 20
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
    
    # Test semantic coherence
    text = """
First paragraph about topic A.
More about topic A.

Second paragraph about topic B.
More about topic B.

Third paragraph about topic C.
More about topic C.
"""
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 1
    assert "topic A" in chunks[0]
    assert "topic B" in chunks[1] if len(chunks) > 2 else True

def test_code_chunker():
    chunker = CodeChunker(chunk_size=100, chunk_overlap=20)
    
    # Test code block preservation
    text = """
Some text before code.

```python
def test_function():
    print("Hello")
    return True
```

Some text after code.
"""
    chunks = chunker.chunk_text(text)
    code_block_found = False
    for chunk in chunks:
        if "```python" in chunk and "def test_function()" in chunk:
            code_block_found = True
            break
    assert code_block_found
    
    # Test indented code block
    text = """
Some text before code.

    def test_function():
        print("Hello")
        return True

Some text after code.
"""
    chunks = chunker.chunk_text(text)
    assert any("def test_function()" in chunk for chunk in chunks)

def test_markdown_chunker():
    chunker = MarkdownChunker(chunk_size=100, chunk_overlap=20)
    
    # Test heading-based splitting
    text = """
# First Section
Content of first section.

## Subsection
Content of subsection.

# Second Section
Content of second section.
"""
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 2
    assert "# First Section" in chunks[0]
    assert "# Second Section" in chunks[-1]
    
    # Test list preservation
    text = """
# Section

- First item
- Second item
  - Subitem 1
  - Subitem 2
- Third item
"""
    chunks = chunker.chunk_text(text)
    list_chunk = None
    for chunk in chunks:
        if "- First item" in chunk:
            list_chunk = chunk
            break
    assert list_chunk is not None
    assert "- Second item" in list_chunk
    assert "- Third item" in list_chunk

def test_recursive_chunker():
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20, max_depth=3)
    
    # Test recursive splitting
    text = """
# Top Level

## Section 1
Content of section 1.
More content.

### Subsection 1.1
Content of subsection 1.1.
More content.

## Section 2
Content of section 2.
More content.

### Subsection 2.1
Content of subsection 2.1.
More content.
"""
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
    
    # Test max depth
    deep_text = "A" * 1000
    chunks = chunker.chunk_text(deep_text)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks) 