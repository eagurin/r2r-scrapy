R2R Scrapy Implementation Guide

Based on the documentation you provided, I'll outline how to implement the R2R Scrapy project - a high-performance tool for asynchronous collection, processing, and indexing of technical documentation for RAG (Retrieval-Augmented Generation) systems.

Project Overview

R2R Scrapy is designed to efficiently collect, process, and index technical documentation from various sources, optimizing it for RAG systems. The project focuses on documentation for libraries, frameworks, and APIs, providing intelligent processing and structuring of data.

Implementation Approach

1. Project Structure

First, let's set up the project structure following the architecture outlined in the documentation:

r2r_scrapy/
├── spiders/                  # Specialized spiders for different documentation types
│   ├── api_spider.py         # Spider for API documentation
│   ├── tutorial_spider.py    # Spider for tutorials and guides
│   ├── github_spider.py      # Spider for GitHub repositories
│   └── blog_spider.py        # Spider for technical blogs
├── processors/               # Content processors
│   ├── code_processor.py     # Code block processing
│   ├── markdown_processor.py # Markdown processing
│   ├── api_processor.py      # API documentation processing
│   └── html_processor.py     # General HTML processing
├── chunkers/                 # Chunking strategies
│   ├── semantic_chunker.py   # Semantic chunking
│   ├── code_chunker.py       # Code-aware chunking
│   ├── markdown_chunker.py   # Markdown-based chunking
│   └── recursive_chunker.py  # Recursive chunking
├── exporters/                # Data exporters
│   ├── r2r_exporter.py       # R2R API exporter
│   └── file_exporter.py      # Local file exporter
├── middleware/               # Scrapy middleware
│   ├── javascript_middleware.py  # JavaScript rendering
│   └── rate_limiter.py       # Intelligent rate limiting
├── utils/                    # Utility functions
│   ├── url_prioritizer.py    # URL prioritization
│   ├── resource_manager.py   # Resource management
│   ├── quality_monitor.py    # Quality monitoring
│   └── version_control.py    # Version control
├── pipelines/                # Processing pipelines
│   ├── preprocessing_pipeline.py  # Preprocessing
│   ├── content_pipeline.py   # Content processing
│   ├── chunking_pipeline.py  # Chunking
│   └── r2r_pipeline.py       # R2R integration
├── integrations/             # External integrations
│   ├── github_integration.py # GitHub API integration
│   ├── stackoverflow_integration.py # Stack Overflow integration
│   └── wikipedia_integration.py # Wikipedia integration
├── security/                 # Security components
│   ├── key_manager.py        # API key management
│   └── secure_logger.py      # Secure logging
├── cli/                      # Command-line interface
│   ├── commands/             # CLI commands
│   └── main.py               # CLI entry point
├── settings.py               # Global settings
├── config.py                 # Configuration management
└── main.py                   # Main entry point
2. Core Components Implementation

2.1. Specialized Spiders

Let's implement the API documentation spider as an example:

# spiders/api_spider.py
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
2.2. Content Processors

Let's implement the code processor:

# processors/code_processor.py
import re
from pygments import lexers, highlight
from pygments.formatters import HtmlFormatter

class CodeProcessor:
    def __init__(self):
        self.formatter = HtmlFormatter()
    
    def process_code(self, code, language=None):
        """Process code blocks with syntax highlighting and formatting"""
        if not code:
            return code
        
        # Clean up code
        code = self.clean_code(code)
        
        # Detect language if not provided
        if not language:
            language = self.detect_language(code)
        
        # Apply syntax highlighting
        try:
            lexer = lexers.get_lexer_by_name(language)
            highlighted_code = highlight(code, lexer, self.formatter)
            return {
                'original': code,
                'highlighted': highlighted_code,
                'language': language
            }
        except Exception:
            # Fallback if language detection fails
            return {
                'original': code,
                'highlighted': code,
                'language': 'text'
            }
    
    def clean_code(self, code):
        """Clean up code by removing unnecessary whitespace and formatting"""
        # Remove excessive blank lines
        code = re.sub(r'\n{3,}', '\n\n', code)
        
        # Normalize indentation
        lines = code.split('\n')
        if lines:
            # Find minimum indentation
            min_indent = float('inf')
            for line in lines:
                if line.strip():  # Skip empty lines
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)
            
            # Remove common indentation
            if min_indent < float('inf'):
                cleaned_lines = []
                for line in lines:
                    if line.strip():
                        cleaned_lines.append(line[min_indent:])
                    else:
                        cleaned_lines.append('')
                code = '\n'.join(cleaned_lines)
        
        return code
    
    def detect_language(self, code):
        """Detect programming language based on code content"""
        try:
            lexer = lexers.guess_lexer(code)
            return lexer.name.lower()
        except Exception:
            # Fallback to simple heuristics
            if re.search(r'def\s+\w+\s*\(|class\s+\w+\s*\(|import\s+\w+', code):
                return 'python'
            elif re.search(r'function\s+\w+\s*\(|const\s+\w+\s*=|var\s+\w+\s*=|let\s+\w+\s*=', code):
                return 'javascript'
            elif re.search(r'public\s+class|private\s+\w+\(|protected\s+\w+\(', code):
                return 'java'
            return 'text'
2.3. Chunking Strategies

Let's implement the semantic chunker:

# chunkers/semantic_chunker.py
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticChunker:
    def __init__(self, chunk_size=800, chunk_overlap=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def chunk_text(self, text):
        """Split text into semantically coherent chunks"""
        if not text:
            return []
        
        # First, split text into paragraphs
        paragraphs = self._split_into_paragraphs(text)
        
        # If text is short enough, return as a single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        # Create initial chunks based on size
        initial_chunks = self._create_initial_chunks(paragraphs)
        
        # Refine chunks based on semantic coherence
        refined_chunks = self._refine_chunks(initial_chunks)
        
        return refined_chunks
    
    def _split_into_paragraphs(self, text):
        """Split text into paragraphs"""
        # Split by double newlines (common paragraph separator)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def _create_initial_chunks(self, paragraphs):
        """Create initial chunks based on size constraints"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph_size = len(paragraph)
            
            # If adding this paragraph exceeds chunk size and we already have content,
            # finish the current chunk and start a new one
            if current_size + paragraph_size > self.chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_size = 0
                overlap_paragraphs = []
                
                # Add paragraphs from the end of the previous chunk for overlap
                for p in reversed(current_chunk):
                    if overlap_size + len(p) <= self.chunk_overlap:
                        overlap_paragraphs.insert(0, p)
                        overlap_size += len(p)
                    else:
                        break
                
                current_chunk = overlap_paragraphs
                current_size = overlap_size
            
            # Add paragraph to current chunk
            current_chunk.append(paragraph)
            current_size += paragraph_size
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _refine_chunks(self, chunks):
        """Refine chunks based on semantic coherence"""
        if len(chunks) <= 1:
            return chunks
        
        # Calculate TF-IDF vectors for chunks
        try:
            tfidf_matrix = self.vectorizer.fit_transform(chunks)
            
            # Calculate similarity between adjacent chunks
            similarities = []
            for i in range(len(chunks) - 1):
                similarity = cosine_similarity(
                    tfidf_matrix[i:i+1], tfidf_matrix[i+1:i+2]
                )[0][0]
                similarities.append(similarity)
            
            # Identify low-similarity boundaries (potential topic changes)
            mean_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            threshold = max(0.1, mean_similarity - std_similarity)
            
            # Merge chunks with high similarity
            refined_chunks = []
            current_chunk = chunks[0]
            
            for i, similarity in enumerate(similarities):
                if similarity < threshold:
                    # Low similarity indicates a topic change, keep chunks separate
                    refined_chunks.append(current_chunk)
                    current_chunk = chunks[i + 1]
                else:
                    # High similarity, merge chunks with proper overlap handling
                    # Find a good breaking point (end of sentence, paragraph, etc.)
                    break_point = self._find_break_point(chunks[i + 1])
                    if break_point > 0:
                        current_chunk += '\n\n' + chunks[i + 1][:break_point]
                        refined_chunks.append(current_chunk)
                        current_chunk = chunks[i + 1][break_point:]
                    else:
                        current_chunk += '\n\n' + chunks[i + 1]
            
            # Add the last chunk
            if current_chunk:
                refined_chunks.append(current_chunk)
            
            return refined_chunks
        except Exception:
            # Fallback to original chunks if TF-IDF processing fails
            return chunks
    
    def _find_break_point(self, text):
        """Find a good breaking point in the text (end of sentence, paragraph)"""
        # Try to find paragraph break
        match = re.search(r'\n\s*\n', text)
        if match:
            return match.start()
        
        # Try to find sentence break
        matches = list(re.finditer(r'[.!?]\s+', text))
        if matches:
            # Find a sentence break near the middle
            middle = len(text) // 2
            closest_match = min(matches, key=lambda m: abs(m.end() - middle))
            return closest_match.end()
        
        return 0
2.4. R2R Exporter

Let's implement the R2R API exporter:

# exporters/r2r_exporter.py
import asyncio
import aiohttp
import json
import hashlib
import uuid
from datetime import datetime

class R2RExporter:
    def __init__(self, api_url, api_key, batch_size=10, max_concurrency=5):
        self.api_url = api_url
        self.api_key = api_key
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.session = None
    
    async def initialize(self):
        """Initialize the exporter"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close the exporter"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def export_documents(self, documents, collection_id=None):
        """Export documents to R2R API"""
        await self.initialize()
        
        # Process documents in batches
        results = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            batch_results = await self.process_batch(batch, collection_id)
            results.extend(batch_results)
        
        return results
    
    async def process_batch(self, batch, collection_id=None):
        """Process a batch of documents"""
        async with self.semaphore:
            try:
                # Prepare data for R2R API
                r2r_data = self.prepare_r2r_data(batch, collection_id)
                
                # Send data to R2R API
                async with self.session.post(
                    f"{self.api_url}/documents",
                    json=r2r_data,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"R2R API error: {response.status} - {error_text}")
                    
                    result = await response.json()
                    return result
            except Exception as e:
                # Handle errors with retry logic
                print(f"Error sending data to R2R: {e}")
                # Implement retry logic here
                return [{"error": str(e), "document": doc} for doc in batch]
    
    def prepare_r2r_data(self, batch, collection_id=None):
        """Prepare data in the format expected by R2R API"""
        documents = []
        
        for item in batch:
            # Generate document ID if not provided
            doc_id = item.get('document_id') or self.generate_document_id(item)
            
            # Determine collection ID
            coll_id = collection_id or self.get_collection_id(item)
            
            # Determine chunking strategy based on content type
            chunk_strategy = self.determine_chunk_strategy(item)
            
            # Create document object
            doc = {
                "content": item['content'],
                "metadata": item.get('metadata', {}),
                "document_id": doc_id,
                "collection_id": coll_id,
                "chunk_strategy": chunk_strategy
            }
            documents.append(doc)
        
        return {"documents": documents}
    
    def generate_document_id(self, item):
        """Generate a unique document ID"""
        url = item.get('url', '')
        content_hash = hashlib.md5(item.get('content', '').encode()).hexdigest()
        timestamp = datetime.now().isoformat()
        
        # Create a unique ID based on URL, content hash, and timestamp
        unique_string = f"{url}:{content_hash}:{timestamp}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, unique_string))
    
    def get_collection_id(self, item):
        """Determine collection ID based on metadata"""
        metadata = item.get('metadata', {})
        library_name = metadata.get('library_name', '')
        version = metadata.get('version', 'latest')
        
        # Create a collection ID based on library name and version
        if library_name:
            return f"{library_name.lower().replace(' ', '_')}_{version}"
        
        # Fallback to a default collection
        return "documentation_collection"
    
    def determine_chunk_strategy(self, item):
        """Determine the best chunking strategy based on content type"""
        metadata = item.get('metadata', {})
        doc_type = metadata.get('doc_type', '')
        
        if doc_type == 'api_reference':
            return "semantic"
        elif doc_type == 'tutorial':
            return "markdown_header"
        elif doc_type == 'code_example':
            return "code_aware"
        
        # Default to semantic chunking
        return "semantic"
    
    async def create_collection(self, collection_id, metadata=None):
        """Create a new collection in R2R"""
        await self.initialize()
        
        try:
            async with self.session.post(
                f"{self.api_url}/collections",
                json={
                    "collection_id": collection_id,
                    "metadata": metadata or {}
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status not in (200, 201):
                    error_text = await response.text()
                    raise Exception(f"R2R API error: {response.status} - {error_text}")
                
                result = await response.json()
                return result
        except Exception as e:
            print(f"Error creating collection: {e}")
            return {"error": str(e)}
2.5. Quality Monitoring

Let's implement the quality monitoring system:

# utils/quality_monitor.py
import logging
import re
import time
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server

class QualityMonitor:
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
2.6. Command-Line Interface

Let's implement the main CLI entry point:

# cli/main.py
import argparse
import asyncio
import os
import yaml
import logging
from r2r_scrapy.spiders import APIDocSpider, TutorialSpider, GitHubSpider, BlogSpider
from r2r_scrapy.exporters import R2RExporter
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

def load_config(config_path):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def setup_logging(log_level):
    """Set up logging configuration"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

async def create_collection(exporter, library, config):
    """Create a collection in R2R if it doesn't exist"""
    collection_id = f"{library.lower().replace(' ', '_')}_{config.get('version', 'latest')}"
    metadata = {
        "library_name": library,
        "version": config.get('version', 'latest'),
        "description": config.get('description', f"{library} documentation"),
        "url": config.get('url', '')
    }
    
    result = await exporter.create_collection(collection_id, metadata)
    return collection_id

def run_spider(spider_cls, config, settings):
    """Run a spider with the given configuration"""
    process = CrawlerProcess(settings)
    
    # Prepare spider arguments
    spider_args = {
        'domain': config.get('domain'),
        'start_urls': config.get('url'),
        'allowed_paths': config.get('allowed_paths')
    }
    
    # Run the spider
    process.crawl(spider_cls, **spider_args)
    process.start()

async def main():
    parser = argparse.ArgumentParser(description='R2R Scrapy - Documentation Collector for RAG Systems')
    parser.add_argument('--library', required=True, help='Library name to scrape')
    parser.add_argument('--url', required=True, help='URL to start scraping from')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--type', choices=['api', 'tutorial', 'github', 'blog'], default='api', 
                        help='Type of documentation to scrape')
    parser.add_argument('--chunking', choices=['semantic', 'code_aware', 'markdown_header', 'recursive'], 
                        default='semantic', help='Chunking strategy to use')
    parser.add_argument('--chunk-size', type=int, default=800, help='Target chunk size')
    parser.add_argument('--chunk-overlap', type=int, default=150, help='Chunk overlap size')
    parser.add_argument('--incremental', action='store_true', help='Perform incremental update')
    parser.add_argument('--monitor', action='store_true', help='Enable quality monitoring')
    parser.add_argument('--log-level', default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger('r2r_scrapy')
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return
    
    # Get R2R API settings
    r2r_config = config.get('r2r', {})
    r2r_api_key = r2r_config.get('api_key') or os.environ.get('R2R_API_KEY')
    r2r_api_url = r2r_config.get('api_url') or os.environ.get('R2R_API_URL')
    
    if not r2r_api_key or not r2r_api_url:
        logger.error("R2R API key and URL must be provided in config or environment variables")
        return
    
    # Initialize R2R exporter
    exporter = R2RExporter(
        api_url=r2r_api_url,
        api_key=r2r_api_key,
        batch_size=r2r_config.get('batch_size', 10),
        max_concurrency=r2r_config.get('max_concurrency', 5)
    )
    
    # Create collection in R2R
    try:
        collection_id = await create_collection(exporter, args.library, config)
        logger.info(f"Collection created or already exists: {collection_id}")
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        return
    
    # Prepare Scrapy settings
    settings = get_project_settings()
    settings.update({
        'LIBRARY_NAME': args.library,
        'COLLECTION_ID': collection_id,
        'CHUNKING_STRATEGY': args.chunking,
        'CHUNK_SIZE': args.chunk_size,
        'CHUNK_OVERLAP': args.chunk_overlap,
        'INCREMENTAL': args.incremental,
        'MONITOR': args.monitor,
        'R2R_API_KEY': r2r_api_key,
        'R2R_API_URL': r2r_api_url,
        'R2R_BATCH_SIZE': r2r_config.get('batch_size', 10),
        'R2R_MAX_CONCURRENCY': r2r_config.get('max_concurrency', 5),
    })
    
    # Select spider based on documentation type
    spider_map = {
        'api': APIDocSpider,
        'tutorial': TutorialSpider,
        'github': GitHubSpider,
        'blog': BlogSpider
    }
    
    spider_cls = spider_map.get(args.type)
    if not spider_cls:
        logger.error(f"Invalid documentation type: {args.type}")
        return
    
    # Run the spider
    try:
        run_spider(spider_cls, {
            'domain': args.url.split('/')[2],
            'url': args.url,
            'allowed_paths': config.get('allowed_paths', '')
        }, settings)
        logger.info(f"Scraping completed for {args.library}")
    except Exception as e:
        logger.error(f"Error running spider: {e}")
    
    # Close the exporter
    await exporter.close()

if __name__ == "__main__":
    asyncio.run(main())
3. Configuration and Settings

Let's implement the configuration management:

# config.py
import os
import yaml
from typing import Dict, Any, Optional

class Config:
    def __init__(self, config_path: Optional[str] = None):
        self.config = {}
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        
        # Override with environment variables
        self.load_from_env()
    
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Error loading configuration from file: {e}")
    
    def load_from_env(self) -> None:
        """Load configuration from environment variables"""
        # R2R Configuration
        if os.environ.get('R2R_API_KEY'):
            self.set_nested_value(['r2r', 'api_key'], os.environ.get('R2R_API_KEY'))
        
        if os.environ.get('R2R_API_URL'):
            self.set_nested_value(['r2r', 'api_url'], os.environ.get('R2R_API_URL'))
        
        if os.environ.get('R2R_BATCH_SIZE'):
            self.set_nested_value(['r2r', 'batch_size'], int(os.environ.get('R2R_BATCH_SIZE')))
        
        if os.environ.get('R2R_MAX_CONCURRENCY'):
            self.set_nested_value(['r2r', 'max_concurrency'], int(os.environ.get('R2R_MAX_CONCURRENCY')))
        
        # Scrapy Configuration
        if os.environ.get('SCRAPY_CONCURRENT_REQUESTS'):
            self.set_nested_value(['scrapy', 'concurrent_requests'], 
                                 int(os.environ.get('SCRAPY_CONCURRENT_REQUESTS')))
        
        if os.environ.get('SCRAPY_CONCURRENT_REQUESTS_PER_DOMAIN'):
            self.set_nested_value(['scrapy', 'concurrent_requests_per_domain'], 
                                 int(os.environ.get('SCRAPY_CONCURRENT_REQUESTS_PER_DOMAIN')))
        
        if os.environ.get('SCRAPY_DOWNLOAD_DELAY'):
            self.set_nested_value(['scrapy', 'download_delay'], 
                                 float(os.environ.get('SCRAPY_DOWNLOAD_DELAY')))
        
        if os.environ.get('SCRAPY_USER_AGENT'):
            self.set_nested_value(['scrapy', 'user_agent'], os.environ.get('SCRAPY_USER_AGENT'))
        
        # Processing Configuration
        if os.environ.get('PROCESSING_DEFAULT_CHUNKING_STRATEGY'):
            self.set_nested_value(['processing', 'default_chunking_strategy'], 
                                 os.environ.get('PROCESSING_DEFAULT_CHUNKING_STRATEGY'))
        
        if os.environ.get('PROCESSING_CHUNK_SIZE'):
            self.set_nested_value(['processing', 'chunk_size'], 
                                 int(os.environ.get('PROCESSING_CHUNK_SIZE')))
        
        if os.environ.get('PROCESSING_CHUNK_OVERLAP'):
            self.set_nested_value(['processing', 'chunk_overlap'], 
                                 int(os.environ.get('PROCESSING_CHUNK_OVERLAP')))
        
        if os.environ.get('PROCESSING_PRESERVE_CODE_BLOCKS'):
            self.set_nested_value(['processing', 'preserve_code_blocks'], 
                                 os.environ.get('PROCESSING_PRESERVE_CODE_BLOCKS').lower() == 'true')
        
        if os.environ.get('PROCESSING_EXTRACT_METADATA'):
            self.set_nested_value(['processing', 'extract_metadata'], 
                                 os.environ.get('PROCESSING_EXTRACT_METADATA').lower() == 'true')
        
        # Monitoring Configuration
        if os.environ.get('MONITORING_ENABLED'):
            self.set_nested_value(['monitoring', 'enabled'], 
                                 os.environ.get('MONITORING_ENABLED').lower() == 'true')
        
        if os.environ.get('MONITORING_PROMETHEUS_PORT'):
            self.set_nested_value(['monitoring', 'prometheus_port'], 
                                 int(os.environ.get('MONITORING_PROMETHEUS_PORT')))
        
        if os.environ.get('MONITORING_QUALITY_THRESHOLD'):
            self.set_nested_value(['monitoring', 'quality_threshold'], 
                                 float(os.environ.get('MONITORING_QUALITY_THRESHOLD')))
        
        if os.environ.get('MONITORING_ALERT_ON_ERROR'):
            self.set_nested_value(['monitoring', 'alert_on_error'], 
                                 os.environ.get('MONITORING_ALERT_ON_ERROR').lower() == 'true')
    
    def set_nested_value(self, keys: list, value: Any) -> None:
        """Set a nested value in the configuration dictionary"""
        current = self.config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the configuration"""
        keys = key.split('.')
        current = self.config
        
        for k in keys:
            if k not in current:
                return default
            current = current[k]
        
        return current
    
    def get_all(self) -> Dict[str, Any]:
        """Get the entire configuration"""
        return self.config
4. Scrapy Settings

Let's implement the Scrapy settings:

# settings.py
import os
from r2r_scrapy.config import Config

# Load configuration
config = Config(os.environ.get('R2R_SCRAPY_CONFIG', 'config.yaml'))

# Scrapy settings
BOT_NAME = 'r2r_scrapy'

SPIDER_MODULES = ['r2r_scrapy.spiders']
NEWSPIDER_MODULE = 'r2r_scrapy.spiders'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests
CONCURRENT_REQUESTS = config.get('scrapy.concurrent_requests', 16)
CONCURRENT_REQUESTS_PER_DOMAIN = config.get('scrapy.concurrent_requests_per_domain', 8)
DOWNLOAD_DELAY = config.get('scrapy.download_delay', 0.5)

# User agent
USER_AGENT = config.get('scrapy.user_agent', 'R2R Scrapy/1.0 (+https://github.com/eagurin/r2r-scrapy)')

# Enable or disable cookies
COOKIES_ENABLED = False

# Configure item pipelines
ITEM_PIPELINES = {
    'r2r_scrapy.pipelines.preprocessing_pipeline.PreprocessingPipeline': 300,
    'r2r_scrapy.pipelines.content_pipeline.ContentPipeline': 400,
    'r2r_scrapy.pipelines.chunking_pipeline.ChunkingPipeline': 500,
    'r2r_scrapy.pipelines.r2r_pipeline.R2RPipeline': 600,
}

# Enable and configure HTTP caching
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 86400  # 1 day
HTTPCACHE_DIR = 'httpcache'
HTTPCACHE_IGNORE_HTTP_CODES = [500, 502, 503, 504, 400, 401, 403, 404]
HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'

# JavaScript rendering settings (if enabled)
if config.get('scrapy.javascript_rendering', False):
    SPLASH_URL = config.get('scrapy.splash_url', 'http://localhost:8050')
    DOWNLOADER_MIDDLEWARES = {
        'scrapy_splash.SplashCookiesMiddleware': 723,
        'scrapy_splash.SplashMiddleware': 725,
        'scrapy.downloadermiddlewares.httpcompression.HttpCompressionMiddleware': 810,
    }
    SPIDER_MIDDLEWARES = {
        'scrapy_splash.SplashDeduplicateArgsMiddleware': 100,
    }
    DUPEFILTER_CLASS = 'scrapy_splash.SplashAwareDupeFilter'

# Monitoring settings
if config.get('monitoring.enabled', False):
    PROMETHEUS_PORT = config.get('monitoring.prometheus_port', 9090)
    QUALITY_THRESHOLD = config.get('monitoring.quality_threshold', 0.8)
    ALERT_ON_ERROR = config.get('monitoring.alert_on_error', True)

# R2R API settings
R2R_API_URL = config.get('r2r.api_url')
R2R_API_KEY = config.get('r2r.api_key')
R2R_BATCH_SIZE = config.get('r2r.batch_size', 10)
R2R_MAX_CONCURRENCY = config.get('r2r.max_concurrency', 5)

# Processing settings
DEFAULT_CHUNKING_STRATEGY = config.get('processing.default_chunking_strategy', 'semantic')
CHUNK_SIZE = config.get('processing.chunk_size', 800)
CHUNK_OVERLAP = config.get('processing.chunk_overlap', 150)
PRESERVE_CODE_BLOCKS = config.get('processing.preserve_code_blocks', True)
EXTRACT_METADATA = config.get('processing.extract_metadata', True)
5. Main Entry Point

Let's implement the main entry point:

# main.py
import sys
import os
from r2r_scrapy.cli.main import main
import asyncio

if __name__ == "__main__":
    # Add the project directory to the Python path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Run the main function
    asyncio.run(main())
Package Setup

To make this a proper Python package, we need to create a
setup.py
file:

# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="r2r-scrapy",
    version="1.0.0",
    author="Evgeny Gurin",
    author_email="e.a.gurin@gmail.com",
    description="Asynchronous Documentation Collector for RAG Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eagurin/r2r-scrapy",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "scrapy>=2.8.0",
        "aiohttp>=3.8.4",
        "asyncio>=3.4.3",
        "pyyaml>=6.0",
        "click>=8.1.3",
        "tqdm>=4.65.0",
        "langchain>=0.0.267",
        "langchain-text-splitters>=0.0.1",
        "beautifulsoup4>=4.12.2",
        "markdown>=3.4.3",
        "html2text>=2020.1.16",
        "nbformat>=5.9.0",
        "openai>=0.27.8",
        "tiktoken>=0.4.0",
        "cryptography>=41.0.3",
        "prometheus-client>=0.17.1",
        "psutil>=5.9.5",
        "gitpython>=3.1.32",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pygments>=2.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.1",
            "sphinx>=7.1.2",
            "sphinx-rtd-theme>=1.3.0",
            "pre-commit>=3.3.3",
        ],
        "js": [
            "scrapy-splash>=0.8.0",
            "playwright>=1.36.0",
        ],
        "distributed": [
            "redis>=4.6.0",
            "scrapy-redis>=0.7.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "r2r-scrapy=r2r_scrapy.main:main",
        ],
    },
)
Usage Examples

Here are some examples of how to use the implemented R2R Scrapy tool:

Basic Usage

# Scrape a library documentation
r2r-scrapy --library fastapi --url https://fastapi.tiangolo.com/

# Specify documentation type
r2r-scrapy --library react --url https://reactjs.org/docs/ --type framework

# Customize chunking
r2r-scrapy --library pandas --url https://pandas.pydata.org/docs/ --chunking semantic
Advanced Usage

# Incremental update of existing collection
r2r-scrapy --library tensorflow --url https://www.tensorflow.org/api_docs/ --incremental

# Full configuration
r2r-scrapy --library scikit-learn --url https://scikit-learn.org/stable/ \
  --chunking semantic --chunk-size 800 --chunk-overlap 150 \
  --max-workers 8 --rate-limit 2 --quality-threshold 0.8 \
  --embedding-model text-embedding-3-small \
  --extract-metadata --preserve-code-blocks \
  --monitor --alert-on-error
Conclusion

This implementation provides a comprehensive framework for asynchronous collection, processing, and indexing of technical documentation for RAG systems. The modular architecture allows for easy extension and customization, while the integration with R2R ensures efficient indexing and retrieval of documentation.

Key features of this implementation include:

Specialized Spiders: Tailored for different documentation sources
Intelligent Content Processing: Advanced handling of code blocks, markdown, and other content types
Adaptive Chunking Strategies: Different chunking methods based on content type
Quality Monitoring: Real-time monitoring of data quality
Seamless R2R Integration: Direct integration with R2R API for efficient indexing
Extensible Architecture: Modular design for easy customization
This implementation serves as a solid foundation for building a powerful documentation collection system for RAG applications.

ты описал все файлы, которые есть в схеме в README.md?
Assistant
Нет, я не описал все файлы, которые упомянуты в архитектурной схеме из README.md. В моем ответе я реализовал только некоторые ключевые компоненты системы, но не все файлы, которые были перечислены в архитектурной схеме.

Вот список файлов из архитектурной схемы, которые я не реализовал полностью:

Не реализованные файлы:

Spiders:

tutorial_spider.py
- Паук для обработки туториалов и руководств
github_spider.py
- Паук для обработки GitHub репозиториев
blog_spider.py
- Паук для обработки технических блогов
Processors:

api_processor.py
- Обработчик API документации
html_processor.py
- Обработчик HTML контента
Chunkers:

code_chunker.py
- Чанкер с учетом кода
markdown_chunker.py
- Чанкер на основе Markdown
recursive_chunker.py
- Рекурсивный чанкер
Exporters:

file_exporter.py
- Экспортер в локальные файлы
Middleware:

javascript_middleware.py
- Middleware для рендеринга JavaScript
rate_limiter.py
- Интеллектуальное ограничение скорости запросов
Utils:

url_prioritizer.py
- Приоритизация URL
resource_manager.py
- Управление ресурсами
version_control.py
- Контроль версий
Pipelines:

preprocessing_pipeline.py
- Пайплайн предобработки
content_pipeline.py
- Пайплайн обработки контента
chunking_pipeline.py
- Пайплайн чанкинга
r2r_pipeline.py
- Пайплайн интеграции с R2R
Integrations:

github_integration.py
- Интеграция с GitHub API
stackoverflow_integration.py
- Интеграция со Stack Overflow
wikipedia_integration.py
- Интеграция с Wikipedia
Security:

key_manager.py
- Управление API ключами
secure_logger.py
- Безопасное логирование
CLI:

Различные команды в директории
commands/
Для полной реализации проекта потребуется разработать все эти компоненты в соответствии с архитектурой, описанной в README.md. Я реализовал только некоторые ключевые компоненты, чтобы продемонстрировать общий подход к разработке такой системы.

Хотите, чтобы я реализовал какие-то конкретные компоненты из этого списка?

продолжай реализовывать оставшиеся
Assistant
Продолжение реализации R2R Scrapy

Продолжим реализацию оставшихся компонентов системы R2R Scrapy согласно архитектуре, описанной в README.md.

Spiders

Tutorial Spider

# spiders/tutorial_spider.py
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
GitHub Spider

# spiders/github_spider.py
import scrapy
import json
import base64
import re
from urllib.parse import urljoin
from r2r_scrapy.processors import MarkdownProcessor, CodeProcessor

class GitHubSpider(scrapy.Spider):
    name = 'github_spider'
    
    def __init__(self, owner=None, repo=None, branch=None, include_readme=True, 
                 include_docs=True, exclude_tests=True, *args, **kwargs):
        super(GitHubSpider, self).__init__(*args, **kwargs)
        self.owner = owner
        self.repo = repo
        self.branch = branch or 'main'
        self.include_readme = self._parse_bool(include_readme)
        self.include_docs = self._parse_bool(include_docs)
        self.exclude_tests = self._parse_bool(exclude_tests)
        
        # GitHub API base URL
        self.api_base = 'https://api.github.com'
        
        # Set GitHub token from settings if available
        self.github_token = kwargs.get('github_token') or self.settings.get('GITHUB_TOKEN')
        
        # Initialize processors
        self.markdown_processor = MarkdownProcessor()
        self.code_processor = CodeProcessor()
    
    def _parse_bool(self, value):
        """Parse boolean values from string or boolean"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', 'yes', '1', 't', 'y')
        return bool(value)
    
    def start_requests(self):
        """Start by fetching repository contents"""
        repo_url = f"{self.api_base}/repos/{self.owner}/{self.repo}/contents"
        if self.branch:
            repo_url += f"?ref={self.branch}"
        
        headers = self._get_headers()
        
        yield scrapy.Request(
            url=repo_url,
            headers=headers,
            callback=self.parse_repo_contents
        )
        
        # Also fetch the README if requested
        if self.include_readme:
            readme_url = f"{self.api_base}/repos/{self.owner}/{self.repo}/readme"
            if self.branch:
                readme_url += f"?ref={self.branch}"
            
            yield scrapy.Request(
                url=readme_url,
                headers=headers,
                callback=self.parse_readme
            )
    
    def _get_headers(self):
        """Get headers for GitHub API requests"""
        headers = {
            'Accept': 'application/vnd.github.v3+json',
        }
        if self.github_token:
            headers['Authorization'] = f"token {self.github_token}"
        return headers
    
    def parse_repo_contents(self, response):
        """Parse repository contents"""
        contents = json.loads(response.body)
        
        # Process each item in the repository
        for item in contents:
            # Skip if it's a test directory and exclude_tests is True
            if self.exclude_tests and self._is_test_path(item['path']):
                continue
            
            if item['type'] == 'dir':
                # Process directory
                yield scrapy.Request(
                    url=item['url'],
                    headers=self._get_headers(),
                    callback=self.parse_repo_contents
                )
            elif item['type'] == 'file':
                # Process file if it's a documentation file or we're including all files
                if self._is_doc_file(item['path']) or not self.include_docs:
                    yield scrapy.Request(
                        url=item['url'],
                        headers=self._get_headers(),
                        callback=self.parse_file,
                        meta={'path': item['path']}
                    )
    
    def parse_readme(self, response):
        """Parse repository README"""
        readme = json.loads(response.body)
        
        # Decode content
        if readme.get('content'):
            content = base64.b64decode(readme['content']).decode('utf-8')
            
            # Process markdown content
            processed_content, metadata = self.markdown_processor.process_markdown(content)
            
            # Create item with processed content
            yield {
                'url': readme['html_url'],
                'path': readme['path'],
                'title': f"{self.owner}/{self.repo} - README",
                'content': processed_content,
                'metadata': {
                    **metadata,
                    'library_name': self.repo,
                    'owner': self.owner,
                    'doc_type': 'readme',
                    'github_url': readme['html_url'],
                }
            }
    
    def parse_file(self, response):
        """Parse a file from the repository"""
        file_data = json.loads(response.body)
        path = response.meta.get('path', file_data.get('path', ''))
        
        # Skip binary files
        if file_data.get('encoding') != 'base64' or not file_data.get('content'):
            return
        
        # Decode content
        content = base64.b64decode(file_data['content']).decode('utf-8', errors='replace')
        
        # Process content based on file type
        file_type = self._get_file_type(path)
        
        if file_type == 'markdown':
            processed_content, metadata = self.markdown_processor.process_markdown(content)
        elif file_type == 'code':
            language = self._detect_language_from_path(path)
            processed_content = self.code_processor.process_code(content, language)
            metadata = {
                'language': language,
                'file_type': 'code',
            }
        else:
            # For other file types, just use the raw content
            processed_content = content
            metadata = {
                'file_type': file_type,
            }
        
        # Create item with processed content
        yield {
            'url': file_data['html_url'],
            'path': path,
            'title': self._get_title_from_path(path),
            'content': processed_content,
            'metadata': {
                **metadata,
                'library_name': self.repo,
                'owner': self.owner,
                'doc_type': 'file',
                'github_url': file_data['html_url'],
                'file_path': path,
            }
        }
    
    def _is_test_path(self, path):
        """Check if a path is a test directory or file"""
        test_patterns = [
            r'/tests?/', r'/tests?$',
            r'_tests?/', r'_tests?$',
            r'/specs?/', r'/specs?$',
            r'_specs?/', r'_specs?$',
            r'test_.*\.py$', r'.*_test\.py$',
            r'spec_.*\.js$', r'.*_spec\.js$',
        ]
        return any(re.search(pattern, path, re.I) for pattern in test_patterns)
    
    def _is_doc_file(self, path):
        """Check if a file is a documentation file"""
        doc_patterns = [
            r'\.md$', r'\.rst$', r'\.txt$',
            r'/docs?/', r'/docs?$',
            r'/documentation/', r'/documentation$',
            r'/examples?/', r'/examples?$',
            r'/tutorials?/', r'/tutorials?$',
            r'/guides?/', r'/guides?$',
            r'README', r'CONTRIBUTING', r'CHANGELOG', r'LICENSE',
        ]
        return any(re.search(pattern, path, re.I) for pattern in doc_patterns)
    
    def _get_file_type(self, path):
        """Determine file type from path"""
        if re.search(r'\.(md|markdown)$', path, re.I):
            return 'markdown'
        elif re.search(r'\.(rst|txt)$', path, re.I):
            return 'text'
        elif re.search(r'\.(py|js|java|c|cpp|cs|go|rb|php|ts|swift|kt|rs|sh|bash)$', path, re.I):
            return 'code'
        elif re.search(r'\.(json|yaml|yml|toml|ini|cfg)$', path, re.I):
            return 'config'
        elif re.search(r'\.(html|htm|xml)$', path, re.I):
            return 'markup'
        else:
            return 'other'
    
    def _detect_language_from_path(self, path):
        """Detect programming language from file path"""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.rs': 'rust',
            '.sh': 'bash',
            '.bash': 'bash',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.html': 'html',
            '.xml': 'xml',
        }
        
        ext = re.search(r'(\.[a-zA-Z0-9]+)$', path)
        if ext:
            return extension_map.get(ext.group(1).lower(), 'text')
        return 'text'
    
    def _get_title_from_path(self, path):
        """Generate a title from file path"""
        # Extract filename
        filename = path.split('/')[-1]
        
        # Remove extension
        title = re.sub(r'\.[^.]+$', '', filename)
        
        # Convert to title case and replace underscores/hyphens with spaces
        title = title.replace('_', ' ').replace('-', ' ').title()
        
        return f"{self.owner}/{self.repo} - {title}"
Blog Spider

# spiders/blog_spider.py
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from datetime import datetime
import re
from r2r_scrapy.processors import HTMLProcessor

class BlogSpider(CrawlSpider):
    name = 'blog_spider'
    
    def __init__(self, domain=None, start_urls=None, allowed_paths=None, 
                 article_css=None, content_css=None, *args, **kwargs):
        super(BlogSpider, self).__init__(*args, **kwargs)
        self.allowed_domains = [domain] if domain else []
        self.start_urls = start_urls.split(',') if start_urls else []
        
        # CSS selectors for articles and content
        self.article_css = article_css or 'article, .post, .blog-post, .entry'
        self.content_css = content_css or '.content, .post-content, .entry-content, article'
        
        # Rules for following blog links
        self.rules = (
            # Rule for blog index pages
            Rule(
                LinkExtractor(
                    allow=r'/(blog|posts|articles)(/page/\d+)?/?$',
                    deny=('search', 'print', 'pdf', 'zip', 'download', 'tag', 'category')
                ),
                follow=True
            ),
            # Rule for blog posts
            Rule(
                LinkExtractor(
                    allow=allowed_paths.split(',') if allowed_paths else (r'/(blog|posts|articles)/[\w-]+/?$',),
                    deny=('search', 'print', 'pdf', 'zip', 'download')
                ),
                callback='parse_blog_post',
                follow=True
            ),
        )
        
        self.html_processor = HTMLProcessor()
    
    def parse_blog_post(self, response):
        """Parse a blog post"""
        # Extract article content
        article = response.css(self.article_css).get() or response.css('body').get()
        
        # Process HTML content
        content, metadata = self.html_processor.process(response, content_css=self.content_css)
        
        # Extract publication date
        pub_date = self.extract_publication_date(response)
        
        # Extract author
        author = self.extract_author(response)
        
        # Extract tags/categories
        tags = self.extract_tags(response)
        
        # Create item with processed content
        yield {
            'url': response.url,
            'title': metadata.get('title') or response.css('title::text').get(),
            'content': content,
            'metadata': {
                **metadata,
                'library_name': self.settings.get('LIBRARY_NAME'),
                'doc_type': 'blog_post',
                'publication_date': pub_date,
                'author': author,
                'tags': tags,
                'reading_time': self.calculate_reading_time(content),
            }
        }
    
    def extract_publication_date(self, response):
        """Extract publication date from blog post"""
        # Try common date selectors
        date_selectors = [
            'time::attr(datetime)',
            'meta[property="article:published_time"]::attr(content)',
            '.date::text', '.published::text', '.post-date::text',
            '.entry-date::text', '.post-meta time::text',
        ]
        
        for selector in date_selectors:
            date_str = response.css(selector).get()
            if date_str:
                try:
                    # Try to parse ISO format
                    return datetime.fromisoformat(date_str.replace('Z', '+00:00')).isoformat()
                except ValueError:
                    # Try to parse with regex
                    date_patterns = [
                        r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
                        r'(\d{2}/\d{2}/\d{4})',  # MM/DD/YYYY
                        r'([A-Z][a-z]+ \d{1,2}, \d{4})',  # Month DD, YYYY
                    ]
                    
                    for pattern in date_patterns:
                        match = re.search(pattern, date_str)
                        if match:
                            try:
                                if '-' in match.group(1):
                                    return datetime.strptime(match.group(1), '%Y-%m-%d').isoformat()
                                elif '/' in match.group(1):
                                    return datetime.strptime(match.group(1), '%m/%d/%Y').isoformat()
                                else:
                                    return datetime.strptime(match.group(1), '%B %d, %Y').isoformat()
                            except ValueError:
                                continue
        
        # If no date found, return current date
        return datetime.now().isoformat()
    
    def extract_author(self, response):
        """Extract author from blog post"""
        # Try common author selectors
        author_selectors = [
            'meta[name="author"]::attr(content)',
            '.author::text', '.post-author::text',
            '.entry-author::text', '.byline::text',
            'a[rel="author"]::text',
        ]
        
        for selector in author_selectors:
            author = response.css(selector).get()
            if author:
                return author.strip()
        
        return 'Unknown'
    
    def extract_tags(self, response):
        """Extract tags/categories from blog post"""
        # Try common tag selectors
        tag_selectors = [
            '.tags a::text', '.categories a::text',
            '.post-tags a::text', '.post-categories a::text',
            '.entry-tags a::text', '.entry-categories a::text',
        ]
        
        for selector in tag_selectors:
            tags = response.css(selector).getall()
            if tags:
                return [tag.strip() for tag in tags if tag.strip()]
        
        return []
    
    def calculate_reading_time(self, content):
        """Calculate estimated reading time in minutes"""
        # Average reading speed: 200-250 words per minute
        words = len(content.split())
        reading_time = max(1, round(words / 225))
        return reading_time
Processors

API Processor

# processors/api_processor.py
import re
from bs4 import BeautifulSoup
from r2r_scrapy.processors.code_processor import CodeProcessor

class APIDocProcessor:
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
HTML Processor

# processors/html_processor.py
import re
from bs4 import BeautifulSoup
import html2text
from r2r_scrapy.processors.code_processor import CodeProcessor

class HTMLProcessor:
    def __init__(self):
        self.code_processor = CodeProcessor()
        self.html2text_converter = html2text.HTML2Text()
        self.html2text_converter.ignore_links = False
        self.html2text_converter.ignore_images = False
        self.html2text_converter.ignore_tables = False
        self.html2text_converter.body_width = 0  # No wrapping
    
    def process(self, response, content_css=None):
        """Process HTML content"""
        # Extract main content
        if content_css:
            html_content = response.css(content_css).get() or response.css('body').get()
        else:
            html_content = response.css('body').get()
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove navigation, sidebars, footers, ads
        for element in soup.select('nav, .sidebar, .navigation, footer, .footer, .menu, .ads, .advertisement'):
            if element:
                element.decompose()
        
        # Process code blocks
        code_blocks = soup.find_all(['pre', 'code'])
        for block in code_blocks:
            code_text = block.get_text()
            language = None
            
            # Try to detect language from class
            if block.get('class'):
                for cls in block.get('class'):
                    if cls.startswith('language-') or cls.startswith('lang-'):
                        language = cls.replace('language-', '').replace('lang-', '')
                        break
            
            # Process code
            processed_code = self.code_processor.process_code(code_text, language)
            
            # Replace original code with processed version if needed
            # For now, we'll keep the original code in the HTML
        
        # Extract metadata
        metadata = self.extract_metadata(response, soup)
        
        # Convert HTML to Markdown
        markdown_content = self.html2text_converter.handle(str(soup))
        
        # Clean up markdown content
        cleaned_content = self.clean_markdown(markdown_content)
        
        return cleaned_content, metadata
    
    def extract_metadata(self, response, soup):
        """Extract metadata from HTML"""
        metadata = {}
        
        # Extract title
        metadata['title'] = soup.title.get_text() if soup.title else response.css('title::text').get()
        
        # Extract meta description
        description = soup.find('meta', attrs={'name': 'description'})
        if description:
            metadata['description'] = description.get('content', '')
        
        # Extract meta keywords
        keywords = soup.find('meta', attrs={'name': 'keywords'})
        if keywords:
            metadata['keywords'] = [k.strip() for k in keywords.get('content', '').split(',')]
        
        # Extract canonical URL
        canonical = soup.find('link', attrs={'rel': 'canonical'})
        if canonical:
            metadata['canonical_url'] = canonical.get('href', '')
        
        # Extract Open Graph metadata
        og_title = soup.find('meta', attrs={'property': 'og:title'})
        if og_title:
            metadata['og_title'] = og_title.get('content', '')
        
        og_description = soup.find('meta', attrs={'property': 'og:description'})
        if og_description:
            metadata['og_description'] = og_description.get('content', '')
        
        og_image = soup.find('meta', attrs={'property': 'og:image'})
        if og_image:
            metadata['og_image'] = og_image.get('content', '')
        
        # Extract headings
        headings = []
        for level in range(1, 7):
            for heading in soup.find_all(f'h{level}'):
                headings.append({
                    'level': level,
                    'text': heading.get_text().strip()
                })
        
        metadata['headings'] = headings
        
        return metadata
    
    def clean_markdown(self, markdown):
        """Clean up markdown content"""
        # Remove excessive newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', markdown)
        
        # Fix code block formatting
        cleaned = re.sub(r'```\s+', '```\n', cleaned)
        cleaned = re.sub(r'\s+```', '\n```', cleaned)
        
        # Fix list formatting
        cleaned = re.sub(r'(\n[*-]\s+[^\n]+)(\n[^\n*-])', r'\1\n\2', cleaned)
        
        return cleaned
Chunkers

Code Chunker

# chunkers/code_chunker.py
import re

class CodeChunker:
    def __init__(self, chunk_size=800, chunk_overlap=150, preserve_code_blocks=True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_code_blocks = preserve_code_blocks
    
    def chunk_text(self, text):
        """Split text into chunks, preserving code blocks"""
        if not text:
            return []
        
        # If text is short enough, return as a single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        # Extract code blocks
        code_blocks = []
        if self.preserve_code_blocks:
            code_blocks = self._extract_code_blocks(text)
        
        # If no code blocks or not preserving them, use simple chunking
        if not code_blocks or not self.preserve_code_blocks:
            return self._simple_chunk(text)
        
        # Replace code blocks with placeholders
        placeholder_text = text
        for i, block in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            placeholder_text = placeholder_text.replace(block, placeholder)
        
        # Chunk the text with placeholders
        chunks = self._simple_chunk(placeholder_text)
        
        # Restore code blocks in chunks
        restored_chunks = []
        for chunk in chunks:
            restored_chunk = chunk
            for i, block in enumerate(code_blocks):
                placeholder = f"__CODE_BLOCK_{i}__"
                if placeholder in restored_chunk:
                    restored_chunk = restored_chunk.replace(placeholder, block)
            restored_chunks.append(restored_chunk)
        
        return restored_chunks
    
    def _extract_code_blocks(self, text):
        """Extract code blocks from text"""
        # Match markdown code blocks
        code_blocks = re.findall(r'```[\w]*\n[\s\S]*?\n```', text)
        
        # Match indented code blocks
        indented_blocks = re.findall(r'(?:^|\n)( {4,}[^\n]+(?:\n {4,}[^\n]+)*)', text)
        code_blocks.extend(indented_blocks)
        
        return code_blocks
    
    def _simple_chunk(self, text):
        """Split text into chunks of specified size with overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            # Determine end position
            end = start + self.chunk_size
            
            # If we're at the end of the text, just add the remaining text
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Find a good breaking point (end of sentence, paragraph, etc.)
            break_point = self._find_break_point(text[start:end])
            
            # If no good breaking point found, just use the chunk size
            if break_point == 0:
                break_point = self.chunk_size
            
            # Add chunk
            chunks.append(text[start:start + break_point])
            
            # Move start position for next chunk, accounting for overlap
            start = start + break_point - self.chunk_overlap
            
            # Ensure we're making progress
            if start <= 0:
                start = break_point
        
        return chunks
    
    def _find_break_point(self, text):
        """Find a good breaking point in the text (end of sentence, paragraph)"""
        # Try to find paragraph break
        match = re.search(r'\n\s*\n', text)
        if match:
            return match.start()
        
        # Try to find sentence break
        matches = list(re.finditer(r'[.!?]\s+', text))
        if matches:
            # Find the last sentence break
            return matches[-1].end()
        
        # Try to find line break
        match = re.search(r'\n', text)
        if match:
            return match.start()
        
        return 0
Markdown Chunker

# chunkers/markdown_chunker.py
import re

class MarkdownChunker:
    def __init__(self, chunk_size=800, chunk_overlap=150, heading_split=True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.heading_split = heading_split
    
    def chunk_text(self, text):
        """Split markdown text into chunks based on headings and size"""
        if not text:
            return []
        
        # If text is short enough, return as a single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        # If heading split is enabled, split by headings first
        if self.heading_split:
            heading_chunks = self._split_by_headings(text)
            
            # Further split large chunks by size
            final_chunks = []
            for chunk in heading_chunks:
                if len(chunk) <= self.chunk_size:
                    final_chunks.append(chunk)
                else:
                    final_chunks.extend(self._split_by_size(chunk))
            
            return final_chunks
        else:
            # Split directly by size
            return self._split_by_size(text)
    
    def _split_by_headings(self, text):
        """Split text by markdown headings"""
        # Match all headings (# to ######)
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+?)(?:\s+#{1,6})?$', re.MULTILINE)
        
        # Find all headings
        headings = list(heading_pattern.finditer(text))
        
        # If no headings found, return the whole text
        if not headings:
            return [text]
        
        # Split text by headings
        chunks = []
        for i, heading in enumerate(headings):
            start = heading.start()
            
            # Determine end position
            if i < len(headings) - 1:
                end = headings[i + 1].start()
            else:
                end = len(text)
            
            # Add chunk
            chunks.append(text[start:end])
        
        # Add text before the first heading if it exists
        if headings[0].start() > 0:
            chunks.insert(0, text[:headings[0].start()])
        
        return chunks
    
    def _split_by_size(self, text):
        """Split text into chunks of specified size with overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            # Determine end position
            end = start + self.chunk_size
            
            # If we're at the end of the text, just add the remaining text
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Find a good breaking point (end of paragraph, sentence, etc.)
            break_point = self._find_break_point(text[start:end])
            
            # If no good breaking point found, just use the chunk size
            if break_point == 0:
                break_point = self.chunk_size
            
            # Add chunk
            chunks.append(text[start:start + break_point])
            
            # Move start position for next chunk, accounting for overlap
            start = start + break_point - self.chunk_overlap
            
            # Ensure we're making progress
            if start <= 0:
                start = break_point
        
        return chunks
    
    def _find_break_point(self, text):
        """Find a good breaking point in markdown text"""
        # Try to find paragraph break
        match = re.search(r'\n\s*\n', text)
        if match:
            return match.start()
        
        # Try to find heading
        match = re.search(r'\n#{1,6}\s+', text)
        if match:
            return match.start()
        
        # Try to find list item
        match = re.search(r'\n[*-]\s+', text)
        if match:
            return match.start()
        
        # Try to find sentence break
        matches = list(re.finditer(r'[.!?]\s+', text))
        if matches:
            # Find the last sentence break
            return matches[-1].end()
        
        # Try to find line break
        match = re.search(r'\n', text)
        if match:
            return match.start()
        
        return 0
Recursive Chunker

# chunkers/recursive_chunker.py
import re

class RecursiveChunker:
    def __init__(self, chunk_size=800, chunk_overlap=150, max_depth=3):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_depth = max_depth
    
    def chunk_text(self, text):
        """Split text into chunks using recursive approach"""
        if not text:
            return []
        
        # If text is short enough, return as a single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        # Start recursive chunking
        return self._recursive_chunk(text, 0)
    
    def _recursive_chunk(self, text, depth):
        """Recursively split text into chunks"""
        # If text is short enough or we've reached max depth, use simple chunking
        if len(text) <= self.chunk_size or depth >= self.max_depth:
            return self._simple_chunk(text)
        
        # Try to split by different delimiters based on depth
        if depth == 0:
            # First level: try to split by double newlines (paragraphs)
            delimiter = r'\n\s*\n'
        elif depth == 1:
            # Second level: try to split by headings or single newlines
            delimiter = r'\n#{1,6}\s+|\n'
        else:
            # Third level: try to split by sentences
            delimiter = r'[.!?]\s+'
        
        # Split text by delimiter
        parts = re.split(delimiter, text)
        
        # If splitting produced only one part, move to next depth
        if len(parts) <= 1:
            return self._recursive_chunk(text, depth + 1)
        
        # Recombine parts that are too small
        combined_parts = []
        current_part = ""
        
        for part in parts:
            # Skip empty parts
            if not part.strip():
                continue
            
            # If adding this part would exceed chunk size, finalize current part
            if len(current_part) + len(part) > self.chunk_size and current_part:
                combined_parts.append(current_part)
                current_part = part
            else:
                # Add delimiter back if not the first part
                if current_part:
                    if depth == 0:
                        current_part += "\n\n"
                    elif depth == 1:
                        current_part += "\n"
                    else:
                        current_part += ". "
                
                current_part += part
        
        # Add the last part if it exists
        if current_part:
            combined_parts.append(current_part)
        
        # If recombining didn't help, move to next depth
        if len(combined_parts) <= 1:
            return self._recursive_chunk(text, depth + 1)
        
        # Process each combined part recursively
        chunks = []
        for part in combined_parts:
            if len(part) <= self.chunk_size:
                chunks.append(part)
            else:
                chunks.extend(self._recursive_chunk(part, depth + 1))
        
        return chunks
    
    def _simple_chunk(self, text):
        """Split text into chunks of specified size with overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            # Determine end position
            end = start + self.chunk_size
            
            # If we're at the end of the text, just add the remaining text
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Find a good breaking point (end of sentence, paragraph, etc.)
            break_point = self._find_break_point(text[start:end])
            
            # If no good breaking point found, just use the chunk size
            if break_point == 0:
                break_point = self.chunk_size
            
            # Add chunk
            chunks.append(text[start:start + break_point])
            
            # Move start position for next chunk, accounting for overlap
            start = start + break_point - self.chunk_overlap
            
            # Ensure we're making progress
            if start <= 0:
                start = break_point
        
        return chunks
    
    def _find_break_point(self, text):
        """Find a good breaking point in the text"""
        # Try to find paragraph break
        match = re.search(r'\n\s*\n', text)
        if match:
            return match.start()
        
        # Try to find heading
        match = re.search(r'\n#{1,6}\s+', text)
        if match:
            return match.start()
        
        # Try to find sentence break
        matches = list(re.finditer(r'[.!?]\s+', text))
        if matches:
            # Find the last sentence break
            return matches[-1].end()
        
        # Try to find line break
        match = re.search(r'\n', text)
        if match:
            return match.start()
        
        return 0
Exporters

File Exporter

# exporters/file_exporter.py
import os
import json
import yaml
import hashlib
from datetime import datetime

class FileExporter:
    def __init__(self, output_dir='./output', format='json'):
        self.output_dir = output_dir
        self.format = format.lower()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for different content types
        self.docs_dir = os.path.join(output_dir, 'docs')
        self.chunks_dir = os.path.join(output_dir, 'chunks')
        self.metadata_dir = os.path.join(output_dir, 'metadata')
        
        os.makedirs(self.docs_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
    
    def export_documents(self, documents, collection_id=None):
        """Export documents to files"""
        results = []
        
        for doc in documents:
            # Generate document ID if not provided
            doc_id = doc.get('document_id') or self._generate_document_id(doc)
            
            # Determine collection directory
            collection_dir = os.path.join(self.docs_dir, collection_id or 'default')
            os.makedirs(collection_dir, exist_ok=True)
            
            # Export document
            result = self._export_document(doc, doc_id, collection_dir)
            results.append(result)
        
        # Export collection metadata
        if collection_id:
            self._export_collection_metadata(collection_id, len(results))
        
        return results
    
    def _export_document(self, document, doc_id, collection_dir):
        """Export a single document"""
        # Prepare document data
        doc_data = {
            'document_id': doc_id,
            'content': document.get('content', ''),
            'metadata': document.get('metadata', {}),
            'url': document.get('url', ''),
            'title': document.get('title', ''),
            'export_timestamp': datetime.now().isoformat(),
        }
        
        # Determine file path
        file_name = f"{doc_id}.{self.format}"
        file_path = os.path.join(collection_dir, file_name)
        
        # Write document to file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if self.format == 'json':
                    json.dump(doc_data, f, ensure_ascii=False, indent=2)
                elif self.format == 'yaml':
                    yaml.dump(doc_data, f, allow_unicode=True)
                else:  # Plain text
                    f.write(doc_data['content'])
            
            # Export metadata separately
            self._export_metadata(doc_data, doc_id)
            
            return {
                'document_id': doc_id,
                'file_path': file_path,
                'success': True,
            }
        except Exception as e:
            return {
                'document_id': doc_id,
                'error': str(e),
                'success': False,
            }
    
    def _export_metadata(self, doc_data, doc_id):
        """Export document metadata separately"""
        metadata = {
            'document_id': doc_id,
            'metadata': doc_data['metadata'],
            'url': doc_data.get('url', ''),
            'title': doc_data.get('title', ''),
            'content_length': len(doc_data.get('content', '')),
            'export_timestamp': doc_data.get('export_timestamp'),
        }
        
        # Write metadata to file
        metadata_path = os.path.join(self.metadata_dir, f"{doc_id}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def _export_collection_metadata(self, collection_id, doc_count):
        """Export collection metadata"""
        metadata = {
            'collection_id': collection_id,
            'document_count': doc_count,
            'export_timestamp': datetime.now().isoformat(),
        }
        
        # Write metadata to file
        metadata_path = os.path.join(self.metadata_dir, f"{collection_id}_collection.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def _generate_document_id(self, document):
        """Generate a unique document ID"""
        url = document.get('url', '')
        title = document.get('title', '')
        content_sample = document.get('content', '')[:1000]  # Use first 1000 chars of content
        timestamp = datetime.now().isoformat()
        
        # Create a unique string
        unique_string = f"{url}:{title}:{content_sample}:{timestamp}"
        
        # Generate MD5 hash
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def export_chunks(self, chunks, doc_id):
        """Export chunks to files"""
        results = []
        
        # Create directory for document chunks
        chunks_doc_dir = os.path.join(self.chunks_dir, doc_id)
        os.makedirs(chunks_doc_dir, exist_ok=True)
        
        for i, chunk in enumerate(chunks):
            # Prepare chunk data
            chunk_data = {
                'document_id': doc_id,
                'chunk_id': f"{doc_id}_{i}",
                'content': chunk,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'export_timestamp': datetime.now().isoformat(),
            }
            
            # Determine file path
            file_name = f"chunk_{i}.{self.format}"
            file_path = os.path.join(chunks_doc_dir, file_name)
            
            # Write chunk to file
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    if self.format == 'json':
                        json.dump(chunk_data, f, ensure_ascii=False, indent=2)
                    elif self.format == 'yaml':
                        yaml.dump(chunk_data, f, allow_unicode=True)
                    else:  # Plain text
                        f.write(chunk_data['content'])
                
                results.append({
                    'chunk_id': chunk_data['chunk_id'],
                    'file_path': file_path,
                    'success': True,
                })
            except Exception as e:
                results.append({
                    'chunk_id': chunk_data['chunk_id'],
                    'error': str(e),
                    'success': False,
                })
        
        return results
Middleware

JavaScript Middleware

# middleware/javascript_middleware.py
from scrapy.http import HtmlResponse
from scrapy.downloadermiddlewares.retry import RetryMiddleware
import logging
import asyncio
from playwright.async_api import async_playwright

class JavaScriptMiddleware:
    """Middleware for rendering JavaScript using Playwright"""
    
    def __init__(self, settings):
        self.logger = logging.getLogger(__name__)
        self.browser = None
        self.context = None
        self.timeout = settings.getint('JAVASCRIPT_TIMEOUT', 30)
        self.wait_until = settings.get('JAVASCRIPT_WAIT_UNTIL', 'networkidle')
        self.enabled = settings.getbool('JAVASCRIPT_ENABLED', False)
        self.browser_type = settings.get('JAVASCRIPT_BROWSER_TYPE', 'chromium')
        self.headless = settings.getbool('JAVASCRIPT_HEADLESS', True)
        self.loop = asyncio.get_event_loop()
    
    @classmethod
    def from_crawler(cls, crawler):
        middleware = cls(crawler.settings)
        crawler.signals.connect(middleware.spider_opened, signal=crawler.signals.spider_opened)
        crawler.signals.connect(middleware.spider_closed, signal=crawler.signals.spider_closed)
        return middleware
    
    async def _start_browser(self):
        """Start Playwright browser"""
        self.logger.info("Starting Playwright browser")
        self.playwright = await async_playwright().start()
        
        if self.browser_type == 'chromium':
            self.browser = await self.playwright.chromium.launch(headless=self.headless)
        elif self.browser_type == 'firefox':
            self.browser = await self.playwright.firefox.launch(headless=self.headless)
        elif self.browser_type == 'webkit':
            self.browser = await self.playwright.webkit.launch(headless=self.headless)
        else:
            self.browser = await self.playwright.chromium.launch(headless=self.headless)
        
        self.context = await self.browser.new_context()
    
    async def _close_browser(self):
        """Close Playwright browser"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
    
    async def _render_page(self, url):
        """Render page with JavaScript"""
        if not self.browser:
            await self._start_browser()
        
        page = await self.context.new_page()
        try:
            response = await page.goto(url, wait_until=self.wait_until, timeout=self.timeout * 1000)
            
            # Wait for content to load
            await page.wait_for_load_state('networkidle')
            
            # Get rendered HTML
            content = await page.content()
            
            return {
                'content': content,
                'status': response.status if response else 200,
                'headers': response.headers if response else {},
                'url': response.url if response else url,
            }
        except Exception as e:
            self.logger.error(f"Error rendering page {url}: {e}")
            return None
        finally:
            await page.close()
    
    def process_request(self, request, spider):
        """Process request with JavaScript rendering if needed"""
        # Skip if JavaScript rendering is disabled or request doesn't need it
        if not self.enabled or request.meta.get('dont_render_js', False):
            return None
        
        # Skip if request method is not GET
        if request.method != 'GET':
            return None
        
        # Render page with JavaScript
        try:
            result = self.loop.run_until_complete(self._render_page(request.url))
            
            if not result:
                # If rendering failed, let Scrapy handle the request normally
                return None
            
            # Create a new response with rendered content
            return HtmlResponse(
                url=result['url'],
                status=result['status'],
                headers=result['headers'],
                body=result['content'].encode('utf-8'),
                encoding='utf-8',
                request=request
            )
        except Exception as e:
            self.logger.error(f"Error in JavaScript middleware: {e}")
            return None
    
    def spider_opened(self, spider):
        """Initialize browser when spider opens"""
        if self.enabled:
            self.loop.run_until_complete(self._start_browser())
    
    def spider_closed(self, spider):
        """Close browser when spider closes"""
        if self.enabled:
            self.loop.run_until_complete(self._close_browser())
Rate Limiter

# middleware/rate_limiter.py
import time
import logging
import random
from scrapy.exceptions import IgnoreRequest
from collections import defaultdict

class RateLimiter:
    """Intelligent rate limiting middleware"""
    
    def __init__(self, settings):
        self.logger = logging.getLogger(__name__)
        
        # Default delay settings
        self.default_delay = settings.getfloat('DOWNLOAD_DELAY', 0)
        self.default_concurrent = settings.getint('CONCURRENT_REQUESTS_PER_DOMAIN', 8)
        
        # Adaptive settings
        self.adaptive = settings.getbool('ADAPTIVE_RATE_LIMITING', True)
        self.min_delay = settings.getfloat('MIN_DOWNLOAD_DELAY', 0.5)
        self.max_delay = settings.getfloat('MAX_DOWNLOAD_DELAY', 5.0)
        self.backoff_factor = settings.getfloat('BACKOFF_FACTOR', 1.5)
        self.success_factor = settings.getfloat('SUCCESS_FACTOR', 0.8)
        
        # Domain-specific settings
        self.domain_delays = {}
        domain_delays = settings.get('DOMAIN_DELAYS', {})
        for domain, delay in domain_delays.items():
            self.domain_delays[domain] = float(delay)
        
        # State tracking
        self.last_request_time = defaultdict(float)
        self.request_count = defaultdict(int)
        self.error_count = defaultdict(int)
        self.success_count = defaultdict(int)
        self.current_delay = defaultdict(lambda: self.default_delay)
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)
    
    def process_request(self, request, spider):
        """Process request with rate limiting"""
        # Extract domain from URL
        domain = self._get_domain(request.url)
        
        # Get current delay for this domain
        delay = self._get_delay(domain)
        
        # Apply jitter to avoid synchronized requests
        jitter = random.uniform(0.8, 1.2)
        delay = delay * jitter
        
        # Check if we need to wait
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time[domain]
        
        if time_since_last_request < delay:
            wait_time = delay - time_since_last_request
            self.logger.debug(f"Rate limiting: waiting {wait_time:.2f}s for {domain}")
            time.sleep(wait_time)
        
        # Update state
        self.last_request_time[domain] = time.time()
        self.request_count[domain] += 1
        
        # Check concurrent request limit
        concurrent_limit = self._get_concurrent_limit(domain)
        if self.request_count[domain] - (self.success_count[domain] + self.error_count[domain]) > concurrent_limit:
            self.logger.debug(f"Too many concurrent requests for {domain}, delaying")
            raise IgnoreRequest(f"Too many concurrent requests for {domain}")
    
    def process_response(self, request, response, spider):
        """Process response to adjust rate limiting"""
        domain = self._get_domain(request.url)
        
        # Update success count
        if 200 <= response.status < 300:
            self.success_count[domain] += 1
            
            # Adjust delay if adaptive rate limiting is enabled
            if self.adaptive:
                # Gradually decrease delay for successful requests
                current_delay = self.current_delay[domain]
                if current_delay > self.min_delay:
                    new_delay = max(self.min_delay, current_delay * self.success_factor)
                    self.current_delay[domain] = new_delay
                    self.logger.debug(f"Decreasing delay for {domain} to {new_delay:.2f}s")
        else:
            # Handle error responses
            self.error_count[domain] += 1
            
            # Adjust delay for error responses
            if self.adaptive:
                # Increase delay for error responses
                current_delay = self.current_delay[domain]
                if response.status in (429, 503):  # Too Many Requests or Service Unavailable
                    # Significant backoff for rate limiting errors
                    new_delay = min(self.max_delay, current_delay * self.backoff_factor * 2)
                else:
                    # Moderate backoff for other errors
                    new_delay = min(self.max_delay, current_delay * self.backoff_factor)
                
                self.current_delay[domain] = new_delay
                self.logger.debug(f"Increasing delay for {domain} to {new_delay:.2f}s due to status {response.status}")
        
        return response
    
    def process_exception(self, request, exception, spider):
        """Process exception to adjust rate limiting"""
        domain = self._get_domain(request.url)
        
        # Update error count
        self.error_count[domain] += 1
        
        # Adjust delay for exceptions
        if self.adaptive:
            current_delay = self.current_delay[domain]
            new_delay = min(self.max_delay, current_delay * self.backoff_factor)
            self.current_delay[domain] = new_delay
            self.logger.debug(f"Increasing delay for {domain} to {new_delay:.2f}s due to exception: {exception}")
    
    def _get_domain(self, url):
        """Extract domain from URL"""
        from urllib.parse import urlparse
        return urlparse(url).netloc
    
    def _get_delay(self, domain):
        """Get delay for a specific domain"""
        # Check if we have a specific delay for this domain
        if domain in self.domain_delays:
            return self.domain_delays[domain]
        
        # Use adaptive delay if enabled
        if self.adaptive:
            return self.current_delay[domain]
        
        # Fall back to default delay
        return self.default_delay
    
    def _get_concurrent_limit(self, domain):
        """Get concurrent request limit for a domain"""
        # Could be extended to have domain-specific limits
        return self.default_concurrent
    
    def get_stats(self):
        """Get rate limiting statistics"""
        stats = {}
        for domain in self.request_count.keys():
            stats[domain] = {
                'requests': self.request_count[domain],
                'successes': self.success_count[domain],
                'errors': self.error_count[domain],
                'current_delay': self.current_delay[domain],
            }
        return stats
Utils

URL Prioritizer

# utils/url_prioritizer.py
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
Resource Manager

# utils/resource_manager.py
import psutil
import time
import logging
import threading
import asyncio
from collections import deque

class ResourceManager:
    """Manage system resources for optimal performance"""
    
    def __init__(self, settings=None):
        self.logger = logging.getLogger(__name__)
        
        # Default settings
        self.settings = settings or {}
        
        # Resource limits
        self.max_cpu_percent = self.settings.get('MAX_CPU_PERCENT', 80)
        self.max_memory_percent = self.settings.get('MAX_MEMORY_PERCENT', 80)
        self.check_interval = self.settings.get('RESOURCE_CHECK_INTERVAL', 5)  # seconds
        
        # Task queue
        self.task_queue = deque()
        self.active_tasks = 0
        self.max_concurrent_tasks = self.settings.get('MAX_CONCURRENT_TASKS', 10)
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Start monitoring thread
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_resources(self):
        """Monitor system resources"""
        while self.running:
            try:
                # Check CPU and memory usage
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                # Log resource usage
                self.logger.debug(f"Resource usage: CPU {cpu_percent}%, Memory {memory_percent}%")
                
                # Adjust max concurrent tasks based on resource usage
                self._adjust_concurrency(cpu_percent, memory_percent)
                
                # Process task queue
                self._process_queue()
                
                # Wait before next check
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in resource monitor: {e}")
                time.sleep(self.check_interval)
    
    def _adjust_concurrency(self, cpu_percent, memory_percent):
        """Adjust max concurrent tasks based on resource usage"""
        with self.lock:
            # Start with default max
            new_max = self.settings.get('MAX_CONCURRENT_TASKS', 10)
            
            # Reduce if CPU usage is high
            if cpu_percent > self.max_cpu_percent:
                cpu_factor = 1 - ((cpu_percent - self.max_cpu_percent) / (100 - self.max_cpu_percent))
                new_max = int(new_max * max(0.5, cpu_factor))
            
            # Reduce if memory usage is high
            if memory_percent > self.max_memory_percent:
                memory_factor = 1 - ((memory_percent - self.max_memory_percent) / (100 - self.max_memory_percent))
                new_max = int(new_max * max(0.5, memory_factor))
            
            # Ensure at least one task can run
            new_max = max(1, new_max)
            
            # Update max concurrent tasks if changed
            if new_max != self.max_concurrent_tasks:
                self.logger.info(f"Adjusting max concurrent tasks from {self.max_concurrent_tasks} to {new_max}")
                self.max_concurrent_tasks = new_max
    
    def _process_queue(self):
        """Process tasks in the queue"""
        with self.lock:
            # Check if we can start more tasks
            while self.task_queue and self.active_tasks < self.max_concurrent_tasks:
                # Get next task
                task, callback = self.task_queue.popleft()
                
                # Increment active tasks
                self.active_tasks += 1
                
                # Start task in a separate thread
                threading.Thread(target=self._run_task, args=(task, callback), daemon=True).start()
    
    def _run_task(self, task, callback):
        """Run a task and call the callback when done"""
        try:
            # Run the task
            if asyncio.iscoroutinefunction(task):
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(task())
                loop.close()
            else:
                result = task()
            
            # Call callback with result
            if callback:
                callback(result)
        except Exception as e:
            self.logger.error(f"Error running task: {e}")
            # Call callback with None result
            if callback:
                callback(None)
        finally:
            # Decrement active tasks
            with self.lock:
                self.active_tasks -= 1
    
    def submit_task(self, task, callback=None):
        """Submit a task to be executed when resources are available"""
        with self.lock:
            # Add task to queue
            self.task_queue.append((task, callback))
            self.logger.debug(f"Task submitted, queue size: {len(self.task_queue)}")
            
            # Try to process queue immediately
            self._process_queue()
    
    def get_stats(self):
        """Get resource manager statistics"""
        with self.lock:
            return {
                'active_tasks': self.active_tasks,
                'queued_tasks': len(self.task_queue),
                'max_concurrent_tasks': self.max_concurrent_tasks,
                'cpu_percent': psutil.cpu_percent(interval=None),
                'memory_percent': psutil.virtual_memory().percent,
            }
    
    def shutdown(self):
        """Shutdown the resource manager"""
        self.running = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
Version Control

# utils/version_control.py
import os
import re
import git
import logging
import hashlib
import json
from datetime import datetime

class VersionControl:
    """Track and manage document versions"""
    
    def __init__(self, storage_dir='./versions'):
        self.logger = logging.getLogger(__name__)
        self.storage_dir = storage_dir
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize Git repository if it doesn't exist
        self.repo_path = os.path.join(storage_dir, 'repo')
        os.makedirs(self.repo_path, exist_ok=True)
        
        try:
            self.repo = git.Repo(self.repo_path)
        except git.exc.InvalidGitRepositoryError:
            self.repo = git.Repo.init(self.repo_path)
            # Create initial commit
            open(os.path.join(self.repo_path, 'README.md'), 'w').write('# Document Version Control\n')
            self.repo.git.add('README.md')
            self.repo.git.commit('-m', 'Initial commit')
    
    def add_document(self, document, doc_id=None):
        """Add or update a document in version control"""
        # Generate document ID if not provided
        if not doc_id:
            doc_id = self._generate_document_id(document)
        
        # Create document path
        doc_path = os.path.join(self.repo_path, f"{doc_id}.json")
        
        # Check if document exists
        is_new = not os.path.exists(doc_path)
        
        # Calculate document hash
        doc_hash = self._calculate_document_hash(document)
        
        # Check if document has changed
        if not is_new:
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    existing_doc = json.load(f)
                existing_hash = existing_doc.get('hash')
                
                if existing_hash == doc_hash:
                    self.logger.debug(f"Document {doc_id} has not changed, skipping version control")
                    return {
                        'document_id': doc_id,
                        'changed': False,
                        'version': existing_doc.get('version', 1),
                    }
            except Exception as e:
                self.logger.error(f"Error reading existing document: {e}")
        
        # Prepare document data with version info
        version_data = {
            'document_id': doc_id,
            'content': document.get('content', ''),
            'metadata': document.get('metadata', {}),
            'url': document.get('url', ''),
            'title': document.get('title', ''),
            'hash': doc_hash,
            'version': 1 if is_new else (existing_doc.get('version', 1) + 1),
            'timestamp': datetime.now().isoformat(),
        }
        
        # Write document to file
        with open(doc_path, 'w', encoding='utf-8') as f:
            json.dump(version_data, f, ensure_ascii=False, indent=2)
        
        # Add to Git
        self.repo.git.add(doc_path)
        
        # Commit changes
        commit_message = f"{'Add' if is_new else 'Update'} document {doc_id} (version {version_data['version']})"
        self.repo.git.commit('-m', commit_message)
        
        return {
            'document_id': doc_id,
            'changed': True,
            'version': version_data['version'],
            'is_new': is_new,
        }
    
    def get_document_history(self, doc_id):
        """Get version history for a document"""
        doc_path = f"{doc_id}.json"
        
        try:
            # Get commit history for the document
            commits = list(self.repo.iter_commits(paths=doc_path))
            
            history = []
            for commit in commits:
                # Get document content at this commit
                try:
                    content = self.repo.git.show(f"{commit.hexsha}:{doc_path}")
                    doc_data = json.loads(content)
                    
                    history.append({
                        'commit_id': commit.hexsha,
                        'version': doc_data.get('version', 1),
                        'timestamp': commit.committed_datetime.isoformat(),
                        'message': commit.message,
                    })
                except Exception as e:
                    self.logger.error(f"Error getting document at commit {commit.hexsha}: {e}")
            
            return history
        except Exception as e:
            self.logger.error(f"Error getting document history: {e}")
            return []
    
    def get_document_version(self, doc_id, version=None, commit_id=None):
        """Get a specific version of a document"""
        doc_path = f"{doc_id}.json"
        
        try:
            if commit_id:
                # Get document at specific commit
                content = self.repo.git.show(f"{commit_id}:{doc_path}")
                return json.loads(content)
            elif version:
                # Find commit with specific version
                commits = list(self.repo.iter_commits(paths=doc_path))
                
                for commit in commits:
                    try:
                        content = self.repo.git.show(f"{commit.hexsha}:{doc_path}")
                        doc_data = json.loads(content)
                        
                        if doc_data.get('version') == version:
                            return doc_data
                    except Exception:
                        continue
                
                raise ValueError(f"Version {version} not found for document {doc_id}")
            else:
                # Get latest version
                with open(os.path.join(self.repo_path, doc_path), 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Error getting document version: {e}")
            return None
    
    def compare_versions(self, doc_id, version1, version2):
        """Compare two versions of a document"""
        doc1 = self.get_document_version(doc_id, version=version1)
        doc2 = self.get_document_version(doc_id, version=version2)
        
        if not doc1 or not doc2:
            return None
        
        # Compare content
        import difflib
        d = difflib.Differ()
        content1 = doc1.get('content', '').splitlines()
        content2 = doc2.get('content', '').splitlines()
        diff = list(d.compare(content1, content2))
        
        return {
            'document_id': doc_id,
            'version1': version1,
            'version2': version2,
            'timestamp1': doc1.get('timestamp'),
            'timestamp2': doc2.get('timestamp'),
            'diff': diff,
            'metadata_changed': doc1.get('metadata') != doc2.get('metadata'),
        }
    
    def _generate_document_id(self, document):
        """Generate a unique document ID"""
        url = document.get('url', '')
        title = document.get('title', '')
        content_sample = document.get('content', '')[:1000]  # Use first 1000 chars of content
        
        # Create a unique string
        unique_string = f"{url}:{title}:{content_sample}"
        
        # Generate MD5 hash
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def _calculate_document_hash(self, document):
        """Calculate a hash for document content"""
        content = document.get('content', '')
        metadata = json.dumps(document.get('metadata', {}), sort_keys=True)
        
        # Create a string to hash
        hash_string = f"{content}:{metadata}"
        
        # Generate MD5 hash
        return hashlib.md5(hash_string.encode()).hexdigest()
Pipelines

Preprocessing Pipeline

# pipelines/preprocessing_pipeline.py
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
Content Pipeline

# pipelines/content_pipeline.py
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
Chunking Pipeline

# pipelines/chunking_pipeline.py
import logging
from r2r_scrapy.chunkers.semantic_chunker import SemanticChunker
from r2r_scrapy.chunkers.code_chunker import CodeChunker
from r2r_scrapy.chunkers.markdown_chunker import MarkdownChunker
from r2r_scrapy.chunkers.recursive_chunker import RecursiveChunker

class ChunkingPipeline:
    """Pipeline for chunking processed content"""
    
    def __init__(self, settings):
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        
        # Initialize chunkers
        self.chunkers = {
            'semantic': SemanticChunker(
                chunk_size=settings.getint('CHUNK_SIZE', 800),
                chunk_overlap=settings.getint('CHUNK_OVERLAP', 150)
            ),
            'code_aware': CodeChunker(
                chunk_size=settings.getint('CHUNK_SIZE', 800),
                chunk_overlap=settings.getint('CHUNK_OVERLAP', 150),
                preserve_code_blocks=settings.getbool('PRESERVE_CODE_BLOCKS', True)
            ),
            'markdown_header': MarkdownChunker(
                chunk_size=settings.getint('CHUNK_SIZE', 800),
                chunk_overlap=settings.getint('CHUNK_OVERLAP', 150),
                heading_split=True
            ),
            'recursive': RecursiveChunker(
                chunk_size=settings.getint('CHUNK_SIZE', 800),
                chunk_overlap=settings.getint('CHUNK_OVERLAP', 150),
                max_depth=3
            )
        }
        
        # Default chunking strategy
        self.default_strategy = settings.get('DEFAULT_CHUNKING_STRATEGY', 'semantic')
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)
    
    def process_item(self, item, spider):
        """Process a scraped item by chunking its content"""
        # Skip if not content processed
        if not item.get('content_processed'):
            return item
        
        # Skip if already chunked
        if item.get('chunked'):
            return item
        
        # Get content
        content = item.get('content', '')
        
        # Skip empty content
        if not content:
            return item
        
        # Determine chunking strategy
        strategy = self._determine_strategy(item)
        
        # Get chunker
        chunker = self.chunkers.get(strategy, self.chunkers[self.default_strategy])
        
        # Chunk content
        chunks = chunker.chunk_text(content)
        
        # Update item
        item['chunks'] = chunks
        item['chunk_count'] = len(chunks)
        item['chunking_strategy'] = strategy
        item['chunked'] = True
        
        # Add chunking info to metadata
        metadata = item.get('metadata', {})
        metadata['chunk_count'] = len(chunks)
        metadata['chunking_strategy'] = strategy
        item['metadata'] = metadata
        
        return item
    
    def _determine_strategy(self, item):
        """Determine the best chunking strategy for this item"""
        # Check if strategy is specified in item
        if 'chunking_strategy' in item:
            strategy = item['chunking_strategy']
            if strategy in self.chunkers:
                return strategy
        
        # Check content type and doc_type
        content_type = item.get('content_type', 'text')
        doc_type = item.get('metadata', {}).get('doc_type', '')
        
        if content_type == 'markdown':
            return 'markdown_header'
        elif doc_type == 'api_reference':
            return 'code_aware'
        elif 'code_blocks' in item.get('metadata', {}):
            return 'code_aware'
        
        # Default to semantic chunking
        return self.default_strategy
R2R Pipeline

# pipelines/r2r_pipeline.py
import logging
import asyncio
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

class R2RPipeline:
    """Pipeline for exporting processed content to R2R"""
    
    def __init__(self, settings):
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        
        # R2R API settings
        self.r2r_api_url = settings.get('R2R_API_URL')
        self.r2r_api_key = settings.get('R2R_API_KEY')
        self.batch_size = settings.getint('R2R_BATCH_SIZE', 10)
        self.max_concurrency = settings.getint('R2R_MAX_CONCURRENCY', 5)
        
        # Initialize exporter
        self.exporter = None
        
        # Batch processing
        self.batch = []
        self.collection_id = settings.get('COLLECTION_ID')
        
        # Event loop
        self.loop = asyncio.get_event_loop()
    
    @classmethod
    def from_crawler(cls, crawler):
        pipeline = cls(crawler.settings)
        crawler.signals.connect(pipeline.spider_opened, signal=crawler.signals.spider_opened)
        crawler.signals.connect(pipeline.spider_closed, signal=crawler.signals.spider_closed)
        return pipeline
    
    def spider_opened(self, spider):
        """Initialize exporter when spider opens"""
        if self.r2r_api_url and self.r2r_api_key:
            self.exporter = R2RExporter(
                api_url=self.r2r_api_url,
                api_key=self.r2r_api_key,
                batch_size=self.batch_size,
                max_concurrency=self.max_concurrency
            )
            self.loop.run_until_complete(self.exporter.initialize())
    
    def spider_closed(self, spider):
        """Process remaining batch and close exporter when spider closes"""
        if self.exporter and self.batch:
            self.logger.info(f"Processing remaining batch of {len(self.batch)} items")
            self.loop.run_until_complete(self._process_batch())
        
        if self.exporter:
            self.loop.run_until_complete(self.exporter.close())
    
    def process_item(self, item, spider):
        """Process a scraped item by adding it to the batch for R2R export"""
        # Skip if not chunked
        if not item.get('chunked'):
            return item
        
        # Skip if no R2R exporter
        if not self.exporter:
            return item
        
        # Add to batch
        self.batch.append(item)
        
        # Process batch if it reaches the batch size
        if len(self.batch) >= self.batch_size:
            self.loop.run_until_complete(self._process_batch())
        
        return item
    
    async def _process_batch(self):
        """Process a batch of items"""
        if not self.batch:
            return
        
        # Prepare documents for R2R
        documents = []
        for item in self.batch:
            # Create a document for each chunk
            for i, chunk in enumerate(item.get('chunks', [])):
                doc = {
                    'content': chunk,
                    'metadata': {
                        **item.get('metadata', {}),
                        'chunk_index': i,
                        'total_chunks': len(item.get('chunks', [])),
                        'url': item.get('url', ''),
                        'title': item.get('title', ''),
                    },
                    'url': item.get('url', ''),
                    'title': f"{item.get('title', '')} (Chunk {i+1}/{len(item.get('chunks', []))})",
                }
                documents.append(doc)
        
        # Export to R2R
        try:
            results = await self.exporter.export_documents(documents, self.collection_id)
            self.logger.info(f"Exported {len(documents)} chunks to R2R")
            
            # Log any errors
            errors = [r for r in results if 'error' in r]
            if errors:
                self.logger.error(f"Errors exporting to R2R: {len(errors)} errors")
                for error in errors[:5]:  # Log first 5 errors
                    self.logger.error(f"R2R export error: {error.get('error')}")
        except Exception as e:
            self.logger.error(f"Error exporting to R2R: {e}")
        
        # Clear batch
        self.batch = []
Integrations

GitHub Integration

# integrations/github_integration.py
import aiohttp
import asyncio
import base64
import json
import logging
from urllib.parse import quote

class GitHubIntegration:
    """Integration with GitHub API for fetching repository content"""
    
    def __init__(self, token=None):
        self.logger = logging.getLogger(__name__)
        self.token = token
        self.api_base = 'https://api.github.com'
        self.session = None
    
    async def initialize(self):
        """Initialize the integration"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close the integration"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def get_repository_contents(self, owner, repo, path='', ref=None):
        """Get contents of a repository directory"""
        await self.initialize()
        
        url = f"{self.api_base}/repos/{owner}/{repo}/contents/{path}"
        if ref:
            url += f"?ref={ref}"
        
        headers = self._get_headers()
        
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    self.logger.error(f"GitHub API error: {response.status} - {error_text}")
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching repository contents: {e}")
            return None
    
    async def get_file_content(self, owner, repo, path, ref=None):
        """Get content of a specific file"""
        await self.initialize()
        
        url = f"{self.api_base}/repos/{owner}/{repo}/contents/{quote(path)}"
        if ref:
            url += f"?ref={ref}"
        
        headers = self._get_headers()
        
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    file_data = await response.json()
                    
                    # Check if it's a file
                    if file_data.get('type') != 'file':
                        self.logger.error(f"Path is not a file: {path}")
                        return None
                    
                    # Decode content
                    if file_data.get('encoding') == 'base64' and file_data.get('content'):
                        content = base64.b64decode(file_data['content']).decode('utf-8', errors='replace')
                        return {
                            'content': content,
                            'path': file_data['path'],
                            'sha': file_data['sha'],
                            'url': file_data['html_url'],
                            'size': file_data['size'],
                        }
                    else:
                        self.logger.error(f"Unsupported encoding or no content: {path}")
                        return None
                else:
                    error_text = await response.text()
                    self.logger.error(f"GitHub API error: {response.status} - {error_text}")
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching file content: {e}")
            return None
    
    async def get_readme(self, owner, repo, ref=None):
        """Get repository README"""
        await self.initialize()
        
        url = f"{self.api_base}/repos/{owner}/{repo}/readme"
        if ref:
            url += f"?ref={ref}"
        
        headers = self._get_headers()
        
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    readme = await response.json()
                    
                    # Decode content
                    if readme.get('encoding') == 'base64' and readme.get('content'):
                        content = base64.b64decode(readme['content']).decode('utf-8', errors='replace')
                        return {
                            'content': content,
                            'path': readme['path'],
                            'sha': readme['sha'],
                            'url': readme['html_url'],
                            'size': readme['size'],
                        }
                    else:
                        self.logger.error(f"Unsupported encoding or no content for README")
                        return None
                else:
                    error_text = await response.text()
                    self.logger.error(f"GitHub API error: {response.status} - {error_text}")
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching README: {e}")
            return None
    
    async def search_code(self, query, owner=None, repo=None, language=None, path=None):
        """Search for code in repositories"""
        await self.initialize()
        
        # Build search query
        search_query = query
        if owner:
            search_query += f" user:{owner}"
        if repo:
            search_query += f" repo:{owner}/{repo}"
        if language:
            search_query += f" language:{language}"
        if path:
            search_query += f" path:{path}"
        
        url = f"{self.api_base}/search/code?q={quote(search_query)}"
        headers = self._get_headers()
        
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    self.logger.error(f"GitHub API error: {response.status} - {error_text}")
                    return None
        except Exception as e:
            self.logger.error(f"Error searching code: {e}")
            return None
    
    async def get_repository_tree(self, owner, repo, ref='main', recursive=True):
        """Get repository file tree"""
        await self.initialize()
        
        url = f"{self.api_base}/repos/{owner}/{repo}/git/trees/{ref}"
        if recursive:
            url += "?recursive=1"
        
        headers = self._get_headers()
        
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    self.logger.error(f"GitHub API error: {response.status} - {error_text}")
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching repository tree: {e}")
            return None
    
    def _get_headers(self):
        """Get headers for GitHub API requests"""
        headers = {
            'Accept': 'application/vnd.github.v3+json',
        }
        if self.token:
            headers['Authorization'] = f"token {self.token}"
        return headers
Stack Overflow Integration

# integrations/stackoverflow_integration.py
import aiohttp
import asyncio
import logging
import html
import re
from urllib.parse import quote

class StackOverflowIntegration:
    """Integration with Stack Overflow API for fetching examples and solutions"""
    
    def __init__(self, api_key=None):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.api_base = 'https://api.stackexchange.com/2.3'
        self.session = None
    
    async def initialize(self):
        """Initialize the integration"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close the integration"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def search_questions(self, query, tags=None, sort='relevance', order='desc', limit=10):
        """Search for questions on Stack Overflow"""
        await self.initialize()
        
        # Build URL
        url = f"{self.api_base}/search/advanced?site=stackoverflow&q={quote(query)}"
        
        if tags:
            if isinstance(tags, list):
                tags = ';'.join(tags)
            url += f"&tagged={quote(tags)}"
        
        url += f"&sort={sort}&order={order}&pagesize={limit}"
        
        if self.api_key:
            url += f"&key={self.api_key}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_questions(data.get('items', []))
                else:
                    error_text = await response.text()
                    self.logger.error(f"Stack Overflow API error: {response.status} - {error_text}")
                    return []
        except Exception as e:
            self.logger.error(f"Error searching Stack Overflow: {e}")
            return []
    
    async def get_question_answers(self, question_id, sort='votes', order='desc'):
        """Get answers for a specific question"""
        await self.initialize()
        
        url = f"{self.api_base}/questions/{question_id}/answers?site=stackoverflow&sort={sort}&order={order}&filter=withbody"
        
        if self.api_key:
            url += f"&key={self.api_key}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_answers(data.get('items', []))
                else:
                    error_text = await response.text()
                    self.logger.error(f"Stack Overflow API error: {response.status} - {error_text}")
                    return []
        except Exception as e:
            self.logger.error(f"Error fetching answers: {e}")
            return []
    
    async def get_question_with_answers(self, question_id):
        """Get a question with its answers"""
        await self.initialize()
        
        # Get question
        question_url = f"{self.api_base}/questions/{question_id}?site=stackoverflow&filter=withbody"
        
        if self.api_key:
            question_url += f"&key={self.api_key}"
        
        try:
            async with self.session.get(question_url) as response:
                if response.status == 200:
                    question_data = await response.json()
                    questions = self._process_questions(question_data.get('items', []))
                    
                    if not questions:
                        return None
                    
                    question = questions[0]
                    
                    # Get answers
                    answers = await self.get_question_answers(question_id)
                    
                    # Combine question and answers
                    question['answers'] = answers
                    
                    return question
                else:
                    error_text = await response.text()
                    self.logger.error(f"Stack Overflow API error: {response.status} - {error_text}")
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching question with answers: {e}")
            return None
    
    async def search_by_tag(self, tag, sort='votes', order='desc', limit=10):
        """Search for questions with a specific tag"""
        await self.initialize()
        
        url = f"{self.api_base}/questions?site=stackoverflow&tagged={quote(tag)}&sort={sort}&order={order}&pagesize={limit}"
        
        if self.api_key:
            url += f"&key={self.api_key}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_questions(data.get('items', []))
                else:
                    error_text = await response.text()
                    self.logger.error(f"Stack Overflow API error: {response.status} - {error_text}")
                    return []
        except Exception as e:
            self.logger.error(f"Error searching by tag: {e}")
            return []
    
    def _process_questions(self, questions):
        """Process question data"""
        processed = []
        
        for q in questions:
            # Unescape HTML entities
            title = html.unescape(q.get('title', ''))
            body = html.unescape(q.get('body', ''))
            
            # Extract code blocks
            code_blocks = self._extract_code_blocks(body)
            
            processed.append({
                'question_id': q.get('question_id'),
                'title': title,
                'body': self._clean_html(body),
                'tags': q.get('tags', []),
                'score': q.get('score', 0),
                'view_count': q.get('view_count', 0),
                'answer_count': q.get('answer_count', 0),
                'is_answered': q.get('is_answered', False),
                'accepted_answer_id': q.get('accepted_answer_id'),
                'creation_date': q.get('creation_date'),
                'link': q.get('link'),
                'code_blocks': code_blocks,
            })
        
        return processed
    
    def _process_answers(self, answers):
        """Process answer data"""
        processed = []
        
        for a in answers:
            # Unescape HTML entities
            body = html.unescape(a.get('body', ''))
            
            # Extract code blocks
            code_blocks = self._extract_code_blocks(body)
            
            processed.append({
                'answer_id': a.get('answer_id'),
                'body': self._clean_html(body),
                'score': a.get('score', 0),
                'is_accepted': a.get('is_accepted', False),
                'creation_date': a.get('creation_date'),
                'link': a.get('link'),
                'code_blocks': code_blocks,
            })
        
        return processed
    
    def _extract_code_blocks(self, html_content):
        """Extract code blocks from HTML content"""
        code_blocks = []
        
        # Find code blocks in <pre><code> tags
        code_pattern = re.compile(r'<pre><code>(.*?)</code></pre>', re.DOTALL)
        for match in code_pattern.finditer(html_content):
            code = match.group(1)
            code = html.unescape(code)
            code_blocks.append(code)
        
        return code_blocks
    
    def _clean_html(self, html_content):
        """Clean HTML content for better readability"""
        # Remove code blocks (we've already extracted them)
        html_content = re.sub(r'<pre><code>.*?</code></pre>', '[CODE BLOCK]', html_content, flags=re.DOTALL)
        
        # Remove HTML tags
        html_content = re.sub(r'<[^>]+>', ' ', html_content)
        
        # Clean up whitespace
        html_content = re.sub(r'\s+', ' ', html_content).strip()
        
        return html_content
Wikipedia Integration

# integrations/wikipedia_integration.py
import aiohttp
import asyncio
import logging
import re
from urllib.parse import quote

class WikipediaIntegration:
    """Integration with Wikipedia API for fetching context information"""
    
    def __init__(self, language='en'):
        self.logger = logging.getLogger(__name__)
        self.language = language
        self.api_base = f"https://{language}.wikipedia.org/w/api.php"
        self.session = None
    
    async def initialize(self):
        """Initialize the integration"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close the integration"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def search(self, query, limit=5):
        """Search for Wikipedia articles"""
        await self.initialize()
        
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'format': 'json',
            'srlimit': limit,
        }
        
        try:
            async with self.session.get(self.api_base, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    search_results = data.get('query', {}).get('search', [])
                    
                    results = []
                    for result in search_results:
                        results.append({
                            'title': result.get('title', ''),
                            'pageid': result.get('pageid', 0),
                            'snippet': self._clean_snippet(result.get('snippet', '')),
                        })
                    
                    return results
                else:
                    error_text = await response.text()
                    self.logger.error(f"Wikipedia API error: {response.status} - {error_text}")
                    return []
        except Exception as e:
            self.logger.error(f"Error searching Wikipedia: {e}")
            return []
    
    async def get_article(self, title=None, pageid=None):
        """Get a Wikipedia article by title or page ID"""
        await self.initialize()
        
        if not title and not pageid:
            self.logger.error("Either title or pageid must be provided")
            return None
        
        params = {
            'action': 'query',
            'prop': 'extracts|info',
            'exintro': 1,  # Only get the introduction
            'explaintext': 1,  # Get plain text
            'inprop': 'url',
            'format': 'json',
        }
        
        if title:
            params['titles'] = title
        else:
            params['pageids'] = pageid
        
        try:
            async with self.session.get(self.api_base, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    pages = data.get('query', {}).get('pages', {})
                    
                    # There should be only one page
                    for page_id, page_data in pages.items():
                        if page_id == '-1':
                            # Page not found
                            return None
                        
                        return {
                            'title': page_data.get('title', ''),
                            'pageid': int(page_id),
                            'extract': page_data.get('extract', ''),
                            'url': page_data.get('fullurl', ''),
                        }
                    
                    return None
                else:
                    error_text = await response.text()
                    self.logger.error(f"Wikipedia API error: {response.status} - {error_text}")
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching Wikipedia article: {e}")
            return None
    
    async def get_article_sections(self, title=None, pageid=None):
        """Get a Wikipedia article with sections"""
        await self.initialize()
        
        if not title and not pageid:
            self.logger.error("Either title or pageid must be provided")
            return None
        
        params = {
            'action': 'parse',
            'prop': 'sections|text',
            'format': 'json',
        }
        
        if title:
            params['page'] = title
        else:
            params['pageid'] = pageid
        
        try:
            async with self.session.get(self.api_base, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'error' in data:
                        self.logger.error(f"Wikipedia API error: {data['error'].get('info', '')}")
                        return None
                    
                    parse_data = data.get('parse', {})
                    
                    # Get sections
                    sections = parse_data.get('sections', [])
                    
                    # Get main text
                    text_html = parse_data.get('text', {}).get('*', '')
                    
                    # Clean HTML
                    main_text = self._clean_html(text_html)
                    
                    return {
                        'title': parse_data.get('title', ''),
                        'pageid': parse_data.get('pageid', 0),
                        'sections': sections,
                        'text': main_text,
                    }
                else:
                    error_text = await response.text()
                    self.logger.error(f"Wikipedia API error: {response.status} - {error_text}")
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching Wikipedia article sections: {e}")
            return None
    
    async def get_article_summary(self, title):
        """Get a summary of a Wikipedia article"""
        await self.initialize()
        
        # Use the summary endpoint
        url = f"https://{self.language}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    return {
                        'title': data.get('title', ''),
                        'extract': data.get('extract', ''),
                        'extract_html': data.get('extract_html', ''),
                        'thumbnail': data.get('thumbnail', {}).get('source') if 'thumbnail' in data else None,
                        'url': data.get('content_urls', {}).get('desktop', {}).get('page') if 'content_urls' in data else None,
                    }
                else:
                    error_text = await response.text()
                    self.logger.error(f"Wikipedia API error: {response.status} - {error_text}")
                    return None
        except Exception as e:
            self.logger.error(f"Error fet
можешь продолжать
можешь продолжать
Assistant
Продолжаю реализацию оставшихся компонентов системы R2R Scrapy:

Integrations (продолжение)

Wikipedia Integration (продолжение)

# integrations/wikipedia_integration.py (продолжение)
            self.logger.error(f"Error fetching Wikipedia article summary: {e}")
            return None
    
    def _clean_snippet(self, snippet):
        """Clean HTML from search snippet"""
        # Remove HTML tags
        snippet = re.sub(r'<[^>]+>', '', snippet)
        return snippet
    
    def _clean_html(self, html_content):
        """Clean HTML content for better readability"""
        # This is a simple implementation, could be improved with BeautifulSoup
        
        # Remove references
        html_content = re.sub(r'<sup class="reference">.*?</sup>', '', html_content)
        
        # Remove edit links
        html_content = re.sub(r'<span class="mw-editsection">.*?</span>', '', html_content)
        
        # Remove HTML tags
        html_content = re.sub(r'<[^>]+>', ' ', html_content)
        
        # Clean up whitespace
        html_content = re.sub(r'\s+', ' ', html_content).strip()
        
        return html_content
Security

Key Manager

# security/key_manager.py
import os
import json
import base64
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class KeyManager:
    """Secure management of API keys and sensitive data"""
    
    def __init__(self, storage_path=None, master_password=None):
        self.logger = logging.getLogger(__name__)
        
        # Storage path for keys
        self.storage_path = storage_path or os.path.join(os.path.expanduser('~'), '.r2r_scrapy', 'keys.json')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        # Master password for encryption
        self.master_password = master_password or os.environ.get('R2R_SCRAPY_MASTER_PASSWORD')
        
        # Initialize encryption key
        self.encryption_key = self._derive_key()
        
        # Load keys
        self.keys = self._load_keys()
    
    def _derive_key(self):
        """Derive encryption key from master password"""
        if not self.master_password:
            # If no master password, generate a random one and store it
            env_key_path = os.path.join(os.path.dirname(self.storage_path), '.env_key')
            
            if os.path.exists(env_key_path):
                # Load existing key
                with open(env_key_path, 'rb') as f:
                    key = f.read()
            else:
                # Generate new key
                key = Fernet.generate_key()
                
                # Save key
                with open(env_key_path, 'wb') as f:
                    f.write(key)
                
                # Set permissions
                os.chmod(env_key_path, 0o600)
            
            return key
        
        # Derive key from master password
        password = self.master_password.encode()
        salt = b'r2r_scrapy_salt'  # Fixed salt, could be improved
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def _load_keys(self):
        """Load keys from storage"""
        if not os.path.exists(self.storage_path):
            return {}
        
        try:
            with open(self.storage_path, 'r') as f:
                encrypted_data = json.load(f)
            
            # Decrypt data
            cipher = Fernet(self.encryption_key)
            decrypted_data = cipher.decrypt(encrypted_data['data'].encode()).decode()
            
            return json.loads(decrypted_data)
        except Exception as e:
            self.logger.error(f"Error loading keys: {e}")
            return {}
    
    def _save_keys(self):
        """Save keys to storage"""
        try:
            # Encrypt data
            cipher = Fernet(self.encryption_key)
            encrypted_data = cipher.encrypt(json.dumps(self.keys).encode()).decode()
            
            # Save to file
            with open(self.storage_path, 'w') as f:
                json.dump({'data': encrypted_data}, f)
            
            # Set permissions
            os.chmod(self.storage_path, 0o600)
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving keys: {e}")
            return False
    
    def get_key(self, key_name):
        """Get a key by name"""
        return self.keys.get(key_name)
    
    def set_key(self, key_name, key_value):
        """Set a key"""
        self.keys[key_name] = key_value
        return self._save_keys()
    
    def delete_key(self, key_name):
        """Delete a key"""
        if key_name in self.keys:
            del self.keys[key_name]
            return self._save_keys()
        return False
    
    def list_keys(self):
        """List all key names"""
        return list(self.keys.keys())
    
    def rotate_master_key(self, new_master_password):
        """Rotate the master encryption key"""
        # Save current keys
        current_keys = self.keys
        
        # Update master password
        self.master_password = new_master_password
        
        # Derive new key
        self.encryption_key = self._derive_key()
        
        # Set keys and save with new encryption
        self.keys = current_keys
        return self._save_keys()
Secure Logger

# security/secure_logger.py
import logging
import re
import json
import os
from datetime import datetime

class SecureLogger:
    """Secure logging with sensitive data masking"""
    
    def __init__(self, log_dir=None, log_level=logging.INFO, mask_sensitive=True):
        self.log_dir = log_dir or os.path.join(os.getcwd(), 'logs')
        self.mask_sensitive = mask_sensitive
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger('r2r_scrapy')
        self.logger.setLevel(log_level)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create file handler
        log_file = os.path.join(self.log_dir, f"r2r_scrapy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        # Set up sensitive data patterns
        self.sensitive_patterns = [
            (r'api_key\s*[=:]\s*["\']([^"\']+)["\']', 'api_key=***'),
            (r'password\s*[=:]\s*["\']([^"\']+)["\']', 'password=***'),
            (r'token\s*[=:]\s*["\']([^"\']+)["\']', 'token=***'),
            (r'secret\s*[=:]\s*["\']([^"\']+)["\']', 'secret=***'),
            (r'auth\s*[=:]\s*["\']([^"\']+)["\']', 'auth=***'),
            (r'bearer\s+([a-zA-Z0-9._-]+)', 'bearer ***'),
        ]
    
    def debug(self, message, *args, **kwargs):
        """Log debug message with sensitive data masking"""
        if self.mask_sensitive:
            message = self._mask_sensitive_data(message)
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message, *args, **kwargs):
        """Log info message with sensitive data masking"""
        if self.mask_sensitive:
            message = self._mask_sensitive_data(message)
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message, *args, **kwargs):
        """Log warning message with sensitive data masking"""
        if self.mask_sensitive:
            message = self._mask_sensitive_data(message)
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message, *args, **kwargs):
        """Log error message with sensitive data masking"""
        if self.mask_sensitive:
            message = self._mask_sensitive_data(message)
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message, *args, **kwargs):
        """Log critical message with sensitive data masking"""
        if self.mask_sensitive:
            message = self._mask_sensitive_data(message)
        self.logger.critical(message, *args, **kwargs)
    
    def _mask_sensitive_data(self, message):
        """Mask sensitive data in log message"""
        if isinstance(message, dict) or isinstance(message, list):
            # Convert to JSON string for masking
            message = json.dumps(message)
            masked_message = self._mask_json(message)
            return masked_message
        
        if not isinstance(message, str):
            # Convert to string
            message = str(message)
        
        # Apply masking patterns
        masked_message = message
        for pattern, replacement in self.sensitive_patterns:
            masked_message = re.sub(pattern, replacement, masked_message, flags=re.IGNORECASE)
        
        return masked_message
    
    def _mask_json(self, json_str):
        """Mask sensitive data in JSON string"""
        try:
            # Parse JSON
            data = json.loads(json_str)
            
            # Mask sensitive fields
            if isinstance(data, dict):
                self._mask_dict(data)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        self._mask_dict(item)
            
            # Convert back to JSON
            return json.dumps(data)
        except Exception:
            # If not valid JSON, treat as string
            return json_str
    
    def _mask_dict(self, data):
        """Recursively mask sensitive fields in dictionary"""
        sensitive_keys = ['api_key', 'password', 'token', 'secret', 'auth', 'authorization']
        
        for key in list(data.keys()):
            # Check if key is sensitive
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                data[key] = '***'
            elif isinstance(data[key], dict):
                self._mask_dict(data[key])
            elif isinstance(data[key], list):
                for item in data[key]:
                    if isinstance(item, dict):
                        self._mask_dict(item)
CLI Commands

Base Command

# cli/commands/base_command.py
import click
import logging
import os
from r2r_scrapy.config import Config

class BaseCommand:
    """Base class for CLI commands"""
    
    def __init__(self, config_path=None):
        # Load configuration
        self.config = Config(config_path or os.environ.get('R2R_SCRAPY_CONFIG', 'config.yaml'))
        
        # Set up logging
        self.logger = logging.getLogger('r2r_scrapy')
    
    def get_r2r_api_settings(self):
        """Get R2R API settings from config"""
        r2r_api_url = self.config.get('r2r.api_url') or os.environ.get('R2R_API_URL')
        r2r_api_key = self.config.get('r2r.api_key') or os.environ.get('R2R_API_KEY')
        
        if not r2r_api_url or not r2r_api_key:
            raise ValueError("R2R API URL and API key must be provided in config or environment variables")
        
        return {
            'api_url': r2r_api_url,
            'api_key': r2r_api_key,
            'batch_size': self.config.get('r2r.batch_size', 10),
            'max_concurrency': self.config.get('r2r.max_concurrency', 5),
        }
    
    def get_scrapy_settings(self):
        """Get Scrapy settings from config"""
        return {
            'concurrent_requests': self.config.get('scrapy.concurrent_requests', 16),
            'concurrent_requests_per_domain': self.config.get('scrapy.concurrent_requests_per_domain', 8),
            'download_delay': self.config.get('scrapy.download_delay', 0.5),
            'user_agent': self.config.get('scrapy.user_agent', 'R2R Scrapy/1.0 (+https://github.com/eagurin/r2r-scrapy)'),
            'javascript_rendering': self.config.get('scrapy.javascript_rendering', False),
            'splash_url': self.config.get('scrapy.splash_url', 'http://localhost:8050'),
        }
    
    def get_processing_settings(self):
        """Get processing settings from config"""
        return {
            'default_chunking_strategy': self.config.get('processing.default_chunking_strategy', 'semantic'),
            'chunk_size': self.config.get('processing.chunk_size', 800),
            'chunk_overlap': self.config.get('processing.chunk_overlap', 150),
            'preserve_code_blocks': self.config.get('processing.preserve_code_blocks', True),
            'extract_metadata': self.config.get('processing.extract_metadata', True),
        }
    
    def get_monitoring_settings(self):
        """Get monitoring settings from config"""
        return {
            'enabled': self.config.get('monitoring.enabled', True),
            'prometheus_port': self.config.get('monitoring.prometheus_port', 9090),
            'quality_threshold': self.config.get('monitoring.quality_threshold', 0.8),
            'alert_on_error': self.config.get('monitoring.alert_on_error', True),
        }
Scrape Command

# cli/commands/scrape_command.py
import click
import asyncio
import os
import logging
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

class ScrapeCommand(BaseCommand):
    """Command for scraping documentation"""
    
    def run(self, library, url, doc_type='api', chunking='semantic', 
            chunk_size=None, chunk_overlap=None, incremental=False, 
            monitor=False, allowed_paths=None, **kwargs):
        """Run the scrape command"""
        self.logger.info(f"Starting scrape for library: {library}, URL: {url}")
        
        # Get settings
        r2r_settings = self.get_r2r_api_settings()
        scrapy_settings = self.get_scrapy_settings()
        processing_settings = self.get_processing_settings()
        
        # Override settings with command-line arguments
        if chunk_size:
            processing_settings['chunk_size'] = chunk_size
        
        if chunk_overlap:
            processing_settings['chunk_overlap'] = chunk_overlap
        
        # Create R2R exporter
        exporter = R2RExporter(
            api_url=r2r_settings['api_url'],
            api_key=r2r_settings['api_key'],
            batch_size=r2r_settings['batch_size'],
            max_concurrency=r2r_settings['max_concurrency']
        )
        
        # Initialize exporter
        loop = asyncio.get_event_loop()
        loop.run_until_complete(exporter.initialize())
        
        # Create or get collection
        collection_id = f"{library.lower().replace(' ', '_')}_{kwargs.get('version', 'latest')}"
        collection_metadata = {
            'library_name': library,
            'version': kwargs.get('version', 'latest'),
            'description': kwargs.get('description', f"{library} documentation"),
            'url': url,
        }
        
        try:
            result = loop.run_until_complete(exporter.create_collection(collection_id, collection_metadata))
            self.logger.info(f"Collection created or already exists: {collection_id}")
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            return
        
        # Prepare Scrapy settings
        settings = get_project_settings()
        settings.update({
            'LIBRARY_NAME': library,
            'COLLECTION_ID': collection_id,
            'CHUNKING_STRATEGY': chunking,
            'CHUNK_SIZE': processing_settings['chunk_size'],
            'CHUNK_OVERLAP': processing_settings['chunk_overlap'],
            'INCREMENTAL': incremental,
            'MONITOR': monitor,
            'R2R_API_KEY': r2r_settings['api_key'],
            'R2R_API_URL': r2r_settings['api_url'],
            'R2R_BATCH_SIZE': r2r_settings['batch_size'],
            'R2R_MAX_CONCURRENCY': r2r_settings['max_concurrency'],
            'CONCURRENT_REQUESTS': scrapy_settings['concurrent_requests'],
            'CONCURRENT_REQUESTS_PER_DOMAIN': scrapy_settings['concurrent_requests_per_domain'],
            'DOWNLOAD_DELAY': scrapy_settings['download_delay'],
            'USER_AGENT': scrapy_settings['user_agent'],
        })
        
        # Enable JavaScript rendering if needed
        if scrapy_settings['javascript_rendering']:
            settings.update({
                'SPLASH_URL': scrapy_settings['splash_url'],
                'DOWNLOADER_MIDDLEWARES': {
                    'scrapy_splash.SplashCookiesMiddleware': 723,
                    'scrapy_splash.SplashMiddleware': 725,
                    'scrapy.downloadermiddlewares.httpcompression.HttpCompressionMiddleware': 810,
                },
                'SPIDER_MIDDLEWARES': {
                    'scrapy_splash.SplashDeduplicateArgsMiddleware': 100,
                },
                'DUPEFILTER_CLASS': 'scrapy_splash.SplashAwareDupeFilter',
            })
        
        # Select spider based on documentation type
        spider_map = {
            'api': 'api_doc_spider',
            'tutorial': 'tutorial_spider',
            'github': 'github_spider',
            'blog': 'blog_spider',
        }
        
        spider_name = spider_map.get(doc_type, 'api_doc_spider')
        
        # Prepare spider arguments
        spider_args = {
            'domain': url.split('/')[2],
            'start_urls': url,
            'allowed_paths': allowed_paths,
        }
        
        # Add GitHub-specific arguments
        if doc_type == 'github':
            # Extract owner and repo from URL
            url_parts = url.strip('/').split('/')
            if 'github.com' in url and len(url_parts) >= 5:
                owner_index = url_parts.index('github.com') + 1
                if len(url_parts) > owner_index:
                    spider_args['owner'] = url_parts[owner_index]
                    if len(url_parts) > owner_index + 1:
                        spider_args['repo'] = url_parts[owner_index + 1]
            
            # Add other GitHub arguments
            spider_args['branch'] = kwargs.get('branch', 'main')
            spider_args['include_readme'] = kwargs.get('include_readme', True)
            spider_args['include_docs'] = kwargs.get('include_docs', True)
            spider_args['exclude_tests'] = kwargs.get('exclude_tests', True)
        
        # Run the spider
        process = CrawlerProcess(settings)
        process.crawl(spider_name, **spider_args)
        process.start()
        
        # Close the exporter
        loop.run_until_complete(exporter.close())
        
        self.logger.info(f"Scraping completed for {library}")

@click.command()
@click.option('--library', required=True, help='Library name to scrape')
@click.option('--url', required=True, help='URL to start scraping from')
@click.option('--type', 'doc_type', type=click.Choice(['api', 'tutorial', 'github', 'blog']), 
              default='api', help='Type of documentation to scrape')
@click.option('--chunking', type=click.Choice(['semantic', 'code_aware', 'markdown_header', 'recursive']), 
              default='semantic', help='Chunking strategy to use')
@click.option('--chunk-size', type=int, help='Target chunk size')
@click.option('--chunk-overlap', type=int, help='Chunk overlap size')
@click.option('--incremental', is_flag=True, help='Perform incremental update')
@click.option('--monitor', is_flag=True, help='Enable quality monitoring')
@click.option('--allowed-paths', help='Comma-separated list of allowed URL paths')
@click.option('--config', help='Path to configuration file')
@click.option('--version', help='Library version')
@click.option('--description', help='Collection description')
@click.option('--branch', help='GitHub branch (for GitHub scraping)')
@click.option('--include-readme/--exclude-readme', default=True, help='Include README (for GitHub scraping)')
@click.option('--include-docs/--exclude-docs', default=True, help='Include docs directory (for GitHub scraping)')
@click.option('--exclude-tests/--include-tests', default=True, help='Exclude tests directory (for GitHub scraping)')
def scrape(**kwargs):
    """Scrape documentation and index it in R2R"""
    command = ScrapeCommand(config_path=kwargs.pop('config', None))
    command.run(**kwargs)
List Collections Command

# cli/commands/list_collections_command.py
import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

class ListCollectionsCommand(BaseCommand):
    """Command for listing R2R collections"""
    
    async def _list_collections(self):
        """List collections in R2R"""
        # Get R2R API settings
        r2r_settings = self.get_r2r_api_settings()
        
        # Create R2R exporter
        exporter = R2RExporter(
            api_url=r2r_settings['api_url'],
            api_key=r2r_settings['api_key'],
            batch_size=r2r_settings['batch_size'],
            max_concurrency=r2r_settings['max_concurrency']
        )
        
        # Initialize exporter
        await exporter.initialize()
        
        try:
            # Call R2R API to list collections
            async with exporter.session.get(
                f"{r2r_settings['api_url']}/collections",
                headers={
                    "Authorization": f"Bearer {r2r_settings['api_key']}",
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"R2R API error: {response.status} - {error_text}")
                    return []
                
                result = await response.json()
                return result.get('collections', [])
        except Exception as e:
            self.logger.error(f"Error listing collections: {e}")
            return []
        finally:
            # Close exporter
            await exporter.close()
    
    def run(self, output_format='table'):
        """Run the list collections command"""
        self.logger.info("Listing R2R collections")
        
        # Get collections
        loop = asyncio.get_event_loop()
        collections = loop.run_until_complete(self._list_collections())
        
        # Display collections
        if not collections:
            click.echo("No collections found")
            return
        
        if output_format == 'json':
            import json
            click.echo(json.dumps(collections, indent=2))
        else:
            # Display as table
            click.echo("\nCollections:")
            click.echo("-" * 80)
            click.echo(f"{'ID':<30} {'Name':<20} {'Document Count':<15} {'Created At'}")
            click.echo("-" * 80)
            
            for collection in collections:
                click.echo(f"{collection.get('collection_id', ''):<30} "
                           f"{collection.get('metadata', {}).get('library_name', ''):<20} "
                           f"{collection.get('document_count', 0):<15} "
                           f"{collection.get('created_at', '')}")

@click.command()
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), 
              default='table', help='Output format')
@click.option('--config', help='Path to configuration file')
def list_collections(output_format, config):
    """List R2R collections"""
    command = ListCollectionsCommand(config_path=config)
    command.run(output_format=output_format)
Create Collection Command

# cli/commands/create_collection_command.py
import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

class CreateCollectionCommand(BaseCommand):
    """Command for creating a new R2R collection"""
    
    async def _create_collection(self, collection_id, name, description, metadata=None):
        """Create a new collection in R2R"""
        # Get R2R API settings
        r2r_settings = self.get_r2r_api_settings()
        
        # Create R2R exporter
        exporter = R2RExporter(
            api_url=r2r_settings['api_url'],
            api_key=r2r_settings['api_key'],
            batch_size=r2r_settings['batch_size'],
            max_concurrency=r2r_settings['max_concurrency']
        )
        
        # Initialize exporter
        await exporter.initialize()
        
        try:
            # Prepare collection metadata
            collection_metadata = metadata or {}
            collection_metadata.update({
                'name': name,
                'description': description,
            })
            
            # Create collection
            result = await exporter.create_collection(collection_id, collection_metadata)
            return result
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            return {'error': str(e)}
        finally:
            # Close exporter
            await exporter.close()
    
    def run(self, name, collection_id=None, description=None, metadata=None):
        """Run the create collection command"""
        # Generate collection ID if not provided
        if not collection_id:
            collection_id = name.lower().replace(' ', '_')
        
        self.logger.info(f"Creating collection: {name} (ID: {collection_id})")
        
        # Create collection
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._create_collection(
            collection_id=collection_id,
            name=name,
            description=description or f"Collection for {name}",
            metadata=metadata
        ))
        
        # Display result
        if 'error' in result:
            click.echo(f"Error creating collection: {result['error']}")
        else:
            click.echo(f"Collection created: {collection_id}")
            click.echo(f"Result: {result}")

@click.command()
@click.option('--name', required=True, help='Collection name')
@click.option('--id', 'collection_id', help='Collection ID (generated from name if not provided)')
@click.option('--description', help='Collection description')
@click.option('--metadata', help='JSON string with additional metadata')
@click.option('--config', help='Path to configuration file')
def create_collection(name, collection_id, description, metadata, config):
    """Create a new R2R collection"""
    # Parse metadata JSON if provided
    metadata_dict = None
    if metadata:
        import json
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            click.echo("Error: Metadata must be a valid JSON string")
            return
    
    command = CreateCollectionCommand(config_path=config)
    command.run(
        name=name,
        collection_id=collection_id,
        description=description,
        metadata=metadata_dict
    )
Delete Collection Command

# cli/commands/delete_collection_command.py
import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

class DeleteCollectionCommand(BaseCommand):
    """Command for deleting an R2R collection"""
    
    async def _delete_collection(self, collection_id):
        """Delete a collection from R2R"""
        # Get R2R API settings
        r2r_settings = self.get_r2r_api_settings()
        
        # Create R2R exporter
        exporter = R2RExporter(
            api_url=r2r_settings['api_url'],
            api_key=r2r_settings['api_key'],
            batch_size=r2r_settings['batch_size'],
            max_concurrency=r2r_settings['max_concurrency']
        )
        
        # Initialize exporter
        await exporter.initialize()
        
        try:
            # Call R2R API to delete collection
            async with exporter.session.delete(
                f"{r2r_settings['api_url']}/collections/{collection_id}",
                headers={
                    "Authorization": f"Bearer {r2r_settings['api_key']}",
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status not in (200, 204):
                    error_text = await response.text()
                    self.logger.error(f"R2R API error: {response.status} - {error_text}")
                    return {'error': error_text}
                
                return {'success': True}
        except Exception as e:
            self.logger.error(f"Error deleting collection: {e}")
            return {'error': str(e)}
        finally:
            # Close exporter
            await exporter.close()
    
    def run(self, collection_id, force=False):
        """Run the delete collection command"""
        self.logger.info(f"Deleting collection: {collection_id}")
        
        # Confirm deletion if not forced
        if not force:
            confirm = click.confirm(f"Are you sure you want to delete collection '{collection_id}'?")
            if not confirm:
                click.echo("Deletion cancelled")
                return
        
        # Delete collection
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._delete_collection(collection_id))
        
        # Display result
        if 'error' in result:
            click.echo(f"Error deleting collection: {result['error']}")
        else:
            click.echo(f"Collection deleted: {collection_id}")

@click.command()
@click.option('--id', 'collection_id', required=True, help='Collection ID to delete')
@click.option('--force', is_flag=True, help='Delete without confirmation')
@click.option('--config', help='Path to configuration file')
def delete_collection(collection_id, force, config):
    """Delete an R2R collection"""
    command = DeleteCollectionCommand(config_path=config)
    command.run(collection_id=collection_id, force=force)
List Documents Command

# cli/commands/list_documents_command.py
import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

class ListDocumentsCommand(BaseCommand):
    """Command for listing documents in an R2R collection"""
    
    async def _list_documents(self, collection_id=None, limit=100, offset=0):
        """List documents in R2R"""
        # Get R2R API settings
        r2r_settings = self.get_r2r_api_settings()
        
        # Create R2R exporter
        exporter = R2RExporter(
            api_url=r2r_settings['api_url'],
            api_key=r2r_settings['api_key'],
            batch_size=r2r_settings['batch_size'],
            max_concurrency=r2r_settings['max_concurrency']
        )
        
        # Initialize exporter
        await exporter.initialize()
        
        try:
            # Build URL
            url = f"{r2r_settings['api_url']}/documents?limit={limit}&offset={offset}"
            if collection_id:
                url += f"&collection_id={collection_id}"
            
            # Call R2R API to list documents
            async with exporter.session.get(
                url,
                headers={
                    "Authorization": f"Bearer {r2r_settings['api_key']}",
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"R2R API error: {response.status} - {error_text}")
                    return []
                
                result = await response.json()
                return result.get('documents', [])
        except Exception as e:
            self.logger.error(f"Error listing documents: {e}")
            return []
        finally:
            # Close exporter
            await exporter.close()
    
    def run(self, collection_id=None, limit=100, offset=0, output_format='table'):
        """Run the list documents command"""
        self.logger.info(f"Listing documents{' in collection: ' + collection_id if collection_id else ''}")
        
        # Get documents
        loop = asyncio.get_event_loop()
        documents = loop.run_until_complete(self._list_documents(
            collection_id=collection_id,
            limit=limit,
            offset=offset
        ))
        
        # Display documents
        if not documents:
            click.echo("No documents found")
            return
        
        if output_format == 'json':
            import json
            click.echo(json.dumps(documents, indent=2))
        else:
            # Display as table
            click.echo("\nDocuments:")
            click.echo("-" * 100)
            click.echo(f"{'ID':<36} {'Title':<40} {'Collection':<20} {'Chunks'}")
            click.echo("-" * 100)
            
            for doc in documents:
                title = doc.get('metadata', {}).get('title', '') or doc.get('document_id', '')
                if len(title) > 38:
                    title = title[:35] + "..."
                
                click.echo(f"{doc.get('document_id', ''):<36} "
                           f"{title:<40} "
                           f"{doc.get('collection_id', ''):<20} "
                           f"{doc.get('chunk_count', 0)}")

@click.command()
@click.option('--collection', 'collection_id', help='Collection ID to list documents from')
@click.option('--limit', type=int, default=100, help='Maximum number of documents to list')
@click.option('--offset', type=int, default=0, help='Offset for pagination')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), 
              default='table', help='Output format')
@click.option('--config', help='Path to configuration file')
def list_documents(collection_id, limit, offset, output_format, config):
    """List documents in an R2R collection"""
    command = ListDocumentsCommand(config_path=config)
    command.run(
        collection_id=collection_id,
        limit=limit,
        offset=offset,
        output_format=output_format
    )
Get Document Command

# cli/commands/get_document_command.py
import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

class GetDocumentCommand(BaseCommand):
    """Command for getting a document from R2R"""
    
    async def _get_document(self, document_id):
        """Get a document from R2R"""
        # Get R2R API settings
        r2r_settings = self.get_r2r_api_settings()
        
        # Create R2R exporter
        exporter = R2RExporter(
            api_url=r2r_settings['api_url'],
            api_key=r2r_settings['api_key'],
            batch_size=r2r_settings['batch_size'],
            max_concurrency=r2r_settings['max_concurrency']
        )
        
        # Initialize exporter
        await exporter.initialize()
        
        try:
            # Call R2R API to get document
            async with exporter.session.get(
                f"{r2r_settings['api_url']}/documents/{document_id}",
                headers={
                    "Authorization": f"Bearer {r2r_settings['api_key']}",
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"R2R API error: {response.status} - {error_text}")
                    return None
                
                result = await response.json()
                return result
        except Exception as e:
            self.logger.error(f"Error getting document: {e}")
            return None
        finally:
            # Close exporter
            await exporter.close()
    
    def run(self, document_id, output_format='json', output_file=None, include_chunks=False):
        """Run the get document command"""
        self.logger.info(f"Getting document: {document_id}")
        
        # Get document
        loop = asyncio.get_event_loop()
        document = loop.run_until_complete(self._get_document(document_id))
        
        # Check if document exists
        if not document:
            click.echo(f"Document not found: {document_id}")
            return
        
        # Format output
        if output_format == 'json':
            import json
            output = json.dumps(document, indent=2)
        elif output_format == 'text':
            # Just output the content
            output = document.get('content', '')
        else:  # summary
            # Output a summary
            metadata = document.get('metadata', {})
            title = metadata.get('title', '') or document_id
            collection_id = document.get('collection_id', '')
            chunk_count = document.get('chunk_count', 0)
            content_preview = document.get('content', '')[:200] + '...' if document.get('content') else ''
            
            output = f"Document: {document_id}\n"
            output += f"Title: {title}\n"
            output += f"Collection: {collection_id}\n"
            output += f"Chunks: {chunk_count}\n"
            output += f"Metadata: {json.dumps(metadata, indent=2)}\n"
            output += f"Content Preview:\n{content_preview}\n"
        
        # Output result
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output)
            click.echo(f"Document saved to: {output_file}")
        else:
            click.echo(output)

@click.command()
@click.option('--id', 'document_id', required=True, help='Document ID to get')
@click.option('--format', 'output_format', type=click.Choice(['json', 'text', 'summary']), 
              default='json', help='Output format')
@click.option('--output', 'output_file', help='Output file path')
@click.option('--include-chunks', is_flag=True, help='Include document chunks')
@click.option('--config', help='Path to configuration file')
def get_document(document_id, output_format, output_file, include_chunks, config):
    """Get a document from R2R"""
    command = GetDocumentCommand(config_path=config)
    command.run(
        document_id=document_id,
        output_format=output_format,
        output_file=output_file,
        include_chunks=include_chunks
    )
Delete Document Command

# cli/commands/delete_document_command.py
import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

class DeleteDocumentCommand(BaseCommand):
    """Command for deleting a document from R2R"""
    
    async def _delete_document(self, document_id):
        """Delete a document from R2R"""
        # Get R2R API settings
        r2r_settings = self.get_r2r_api_settings()
        
        # Create R2R exporter
        exporter = R2RExporter(
            api_url=r2r_settings['api_url'],
            api_key=r2r_settings['api_key'],
            batch_size=r2r_settings['batch_size'],
            max_concurrency=r2r_settings['max_concurrency']
        )
        
        # Initialize exporter
        await exporter.initialize()
        
        try:
            # Call R2R API to delete document
            async with exporter.session.delete(
                f"{r2r_settings['api_url']}/documents/{document_id}",
                headers={
                    "Authorization": f"Bearer {r2r_settings['api_key']}",
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status not in (200, 204):
                    error_text = await response.text()
                    self.logger.error(f"R2R API error: {response.status} - {error_text}")
                    return {'error': error_text}
                
                return {'success': True}
        except Exception as e:
            self.logger.error(f"Error deleting document: {e}")
            return {'error': str(e)}
        finally:
            # Close exporter
            await exporter.close()
    
    def run(self, document_id, force=False):
        """Run the delete document command"""
        self.logger.info(f"Deleting document: {document_id}")
        
        # Confirm deletion if not forced
        if not force:
            confirm = click.confirm(f"Are you sure you want to delete document '{document_id}'?")
            if not confirm:
                click.echo("Deletion cancelled")
                return
        
        # Delete document
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._delete_document(document_id))
        
        # Display result
        if 'error' in result:
            click.echo(f"Error deleting document: {result['error']}")
        else:
            click.echo(f"Document deleted: {document_id}")

@click.command()
@click.option('--id', 'document_id', required=True, help='Document ID to delete')
@click.option('--force', is_flag=True, help='Delete without confirmation')
@click.option('--config', help='Path to configuration file')
def delete_document(document_id, force, config):
    """Delete a document from R2R"""
    command = DeleteDocumentCommand(config_path=config)
    command.run(document_id=document_id, force=force)
Generate Report Command

# cli/commands/generate_report_command.py
import click
import asyncio
import os
import logging
import json
from datetime import datetime
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

class GenerateReportCommand(BaseCommand):
    """Command for generating a quality report"""
    
    async def _get_collections(self):
        """Get all collections from R2R"""
        # Get R2R API settings
        r2r_settings = self.get_r2r_api_settings()
        
        # Create R2R exporter
        exporter = R2RExporter(
            api_url=r2r_settings['api_url'],
            api_key=r2r_settings['api_key'],
            batch_size=r2r_settings['batch_size'],
            max_concurrency=r2r_settings['max_concurrency']
        )
        
        # Initialize exporter
        await exporter.initialize()
        
        try:
            # Call R2R API to list collections
            async with exporter.session.get(
                f"{r2r_settings['api_url']}/collections",
                headers={
                    "Authorization": f"Bearer {r2r_settings['api_key']}",
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"R2R API error: {response.status} - {error_text}")
                    return []
                
                result = await response.json()
                return result.get('collections', [])
        except Exception as e:
            self.logger.error(f"Error getting collections: {e}")
            return []
        finally:
            # Close exporter
            await exporter.close()
    
    async def _get_documents(self, collection_id=None, limit=1000):
        """Get documents from R2R"""
        # Get R2R API settings
        r2r_settings = self.get_r2r_api_settings()
        
        # Create R2R exporter
        exporter = R2RExporter(
            api_url=r2r_settings['api_url'],
            api_key=r2r_settings['api_key'],
            batch_size=r2r_settings['batch_size'],
            max_concurrency=r2r_settings['max_concurrency']
        )
        
        # Initialize exporter
        await exporter.initialize()
        
        try:
            # Build URL
            url = f"{r2r_settings['api_url']}/documents?limit={limit}"
            if collection_id:
                url += f"&collection_id={collection_id}"
            
            # Call R2R API to list documents
            async with exporter.session.get(
                url,
                headers={
                    "Authorization": f"Bearer {r2r_settings['api_key']}",
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"R2R API error: {response.status} - {error_text}")
                    return []
                
                result = await response.json()
                return result.get('documents', [])
        except Exception as e:
            self.logger.error(f"Error getting documents: {e}")
            return []
        finally:
            # Close exporter
            await exporter.close()
    
    def _generate_html_report(self, collections, documents):
        """Generate HTML report"""
        # Calculate statistics
        total_collections = len(collections)
        total_documents = len(documents)
        total_chunks = sum(doc.get('chunk_count', 0) for doc in documents)
        
        # Count documents by type
        doc_types = {}
        for doc in documents:
            doc_type = doc.get('metadata', {}).get('doc_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        # Count documents by chunking strategy
        chunking_strategies = {}
        for doc in documents:
            strategy = doc.get('metadata', {}).get('chunking_strategy', 'unknown')
            chunking_strategies[strategy] = chunking_strategies.get(strategy, 0) + 1
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>R2R Scrapy Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .summary {{ background-color: #e9f7ef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .chart {{ height: 200px; margin-bottom: 20px; }}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <h1>R2R Scrapy Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Collections: {total_collections}</p>
                <p>Total Documents: {total_documents}</p>
                <p>Total Chunks: {total_chunks}</p>
                <p>Average Chunks per Document: {total_chunks / total_documents if total_documents else 0:.2f}</p>
            </div>
            
            <h2>Document Types</h2>
            <div class="chart">
                <canvas id="docTypesChart"></canvas>
            </div>
            
            <h2>Chunking Strategies</h2>
            <div class="chart">
                <canvas id="chunkingChart"></canvas>
            </div>
            
            <h2>Collections</h2>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Documents</th>
                    <th>Created At</th>
                </tr>
        """
        
        for collection in collections:
            html += f"""
                <tr>
                    <td>{collection.get('collection_id', '')}</td>
                    <td>{collection.get('metadata', {}).get('library_name', '')}</td>
                    <td>{collection.get('document_count', 0)}</td>
                    <td>{collection.get('created_at', '')}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Recent Documents</h2>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Title</th>
                    <th>Collection</th>
                    <th>Chunks</th>
                    <th>Type</th>
                </tr>
        """
        
        # Show only the most recent 20 documents
        for doc in documents[:20]:
            title = doc.get('metadata', {}).get('title', '') or doc.get('document_id', '')
            if len(title) > 40:
                title = title[:37] + "..."
            
            html += f"""
                <tr>
                    <td>{doc.get('document_id', '')}</td>
                    <td>{title}</td>
                    <td>{doc.get('collection_id', '')}</td>
                    <td>{doc.get('chunk_count', 0)}</td>
                    <td>{doc.get('metadata', {}).get('doc_type', 'unknown')}</td>
                </tr>
            """
        
        html += f"""
            </table>
            
            <script>
                // Document Types Chart
                var docTypesCtx = document.getElementById('docTypesChart').getContext('2d');
                var docTypesChart = new Chart(docTypesCtx, {{
                    type: 'pie',
                    data: {{
                        labels: {json.dumps(list(doc_types.keys()))},
                        datasets: [{{
                            data: {json.dumps(list(doc_types.values()))},
                            backgroundColor: [
                                '#4e73df', '#1cc88a', '#36b9cc', '#f6c23e', '#e74a3b',
                                '#858796', '#5a5c69', '#6610f2', '#6f42c1', '#fd7e14'
                            ]
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        legend: {{
                            position: 'right'
                        }}
                    }}
                }});
                
                // Chunking Strategies Chart
                var chunkingCtx = document.getElementById('chunkingChart').getContext('2d');
                var chunkingChart = new Chart(chunkingCtx, {{
                    type: 'pie',
                    data: {{
                        labels: {json.dumps(list(chunking_strategies.keys()))},
                        datasets: [{{
                            data: {json.dumps(list(chunking_strategies.values()))},
                            backgroundColor: [
                                '#1cc88a', '#4e73df', '#36b9cc', '#f6c23e', '#e74a3b',
                                '#858796', '#5a5c69', '#6610f2', '#6f42c1', '#fd7e14'
                            ]
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        legend: {{
                            position: 'right'
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
        """
        
        return html
    
    def _generate_json_report(self, collections, documents):
        """Generate JSON report"""
        # Calculate statistics
        total_collections = len(collections)
        total_documents = len(documents)
        total_chunks = sum(doc.get('chunk_count', 0) for doc in documents)
        
        # Count documents by type
        doc_types = {}
        for doc in documents:
            doc_type = doc.get('metadata', {}).get('doc_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        # Count documents by chunking strategy
        chunking_strategies = {}
        for doc in documents:
            strategy = doc.get('metadata', {}).get('chunking_strategy', 'unknown')
            chunking_strategies[strategy] = chunking_strategies.get(strategy, 0) + 1
        
        # Generate report
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_collections': total_collections,
                'total_documents': total_documents,
                'total_chunks': total_chunks,
                'avg_chunks_per_document': total_chunks / total_documents if total_documents else 0,
            },
            'document_types': doc_types,
            'chunking_strategies': chunking_strategies,
            'collections': collections,
            'recent_documents': documents[:20],  # Only include the 20 most recent documents
        }
        
        return json.dumps(report, indent=2)
    
    def run(self, output_format='html', output_file=None, collection_id=None):
        """Run the generate report command"""
        self.logger.info("Generating quality report")
        
        # Get collections and documents
        loop = asyncio.get_event_loop()
        collections = loop.run_until_complete(self._get_collections())
        documents = loop.run_until_complete(self._get_documents(collection_id))
        
        # Sort documents by creation date (newest first)
        documents.sort(key=lambda d: d.get('created_at', ''), reverse=True)
        
        # Generate report
        if output_format == 'html':
            report = self._generate_html_report(collections, documents)
        else:  # json
            report = self._generate_json_report(collections, documents)
        
        # Output report
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            click.echo(f"Report saved to: {output_file}")
        else:
            click.echo(report)

@click.command()
@click.option('--format', 'output_format', type=click.Choice(['html', 'json']), 
              default='html', help='Output format')
@click.option('--output', 'output_file', help='Output file path')
@click.option('--collection', 'collection_id', help='Filter by collection ID')
@click.option('--config', help='Path to configuration file')
def generate_report(output_format, output_file, collection_id, config):
    """Generate a quality report"""
    command = GenerateReportCommand(config_path=config)
    command.run(
        output_format=output_format,
        output_file=output_file,
        collection_id=collection_id
    )
Main CLI Entry Point

# cli/main.py
import click
from r2r_scrapy.cli.commands.scrape_command import scrape
from r2r_scrapy.cli.commands.list_collections_command import list_collections
from r2r_scrapy.cli.commands.create_collection_command import create_collection
from r2r_scrapy.cli.commands.delete_collection_command import delete_collection
from r2r_scrapy.cli.commands.list_documents_command import list_documents
from r2r_scrapy.cli.commands.get_document_command import get_document
from r2r_scrapy.cli.commands.delete_document_command import delete_document
from r2r_scrapy.cli.commands.generate_report_command import generate_report

@click.group()
@click.version_option(version='1.0.0')
def cli():
    """R2R Scrapy - Asynchronous Documentation Collector for RAG Systems"""
    pass

# Add commands
cli.add_command(scrape)
cli.add_command(list_collections)
cli.add_command(create_collection)
cli.add_command(delete_collection)
cli.add_command(list_documents)
cli.add_command(get_document)
cli.add_command(delete_document)
cli.add_command(generate_report)

if __name__ == '__main__':
    cli()
Заключение

Теперь мы реализовали все основные компоненты системы R2R Scrapy согласно архитектуре, описанной в README.md. Система включает в себя:

Специализированные пауки (Spiders) для различных типов документации:

API документация
Туториалы и руководства
GitHub репозитории
Технические блоги
Процессоры контента (Processors) для обработки различных типов контента:

Обработка кода
Обработка Markdown
Обработка API документации
Обработка HTML
Стратегии чанкинга (Chunkers) для разбиения контента на оптимальные фрагменты:

Семантический чанкинг
Чанкинг с учетом кода
Чанкинг на основе Markdown
Рекурсивный чанкинг
Экспортеры (Exporters) для сохранения данных:

Экспорт в R2R API
Экспорт в локальные файлы
Middleware для расширения функциональности Scrapy:

Рендеринг JavaScript
Интеллекту
