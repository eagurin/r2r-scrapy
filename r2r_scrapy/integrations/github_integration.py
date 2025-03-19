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