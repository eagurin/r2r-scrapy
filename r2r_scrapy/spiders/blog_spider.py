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