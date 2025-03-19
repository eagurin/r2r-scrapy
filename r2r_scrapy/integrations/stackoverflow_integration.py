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