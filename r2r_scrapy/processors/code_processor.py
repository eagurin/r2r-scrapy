import re
from pygments import lexers, highlight
from pygments.formatters import HtmlFormatter

class CodeProcessor:
    """Process code blocks with syntax highlighting and formatting"""
    
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