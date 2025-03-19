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