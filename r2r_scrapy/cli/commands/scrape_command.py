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