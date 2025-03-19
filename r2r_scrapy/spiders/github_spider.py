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