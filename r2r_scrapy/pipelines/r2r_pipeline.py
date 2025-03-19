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