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