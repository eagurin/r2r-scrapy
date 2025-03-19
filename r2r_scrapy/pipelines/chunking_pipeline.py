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