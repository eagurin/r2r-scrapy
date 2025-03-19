import re

class CodeChunker:
    """Split text into chunks while preserving code blocks"""
    
    def __init__(self, chunk_size=800, chunk_overlap=150, preserve_code_blocks=True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_code_blocks = preserve_code_blocks
    
    def chunk_text(self, text):
        """Split text into chunks, preserving code blocks"""
        if not text:
            return []
        
        # If text is short enough, return as a single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        # Extract code blocks
        code_blocks = []
        if self.preserve_code_blocks:
            code_blocks = self._extract_code_blocks(text)
        
        # If no code blocks or not preserving them, use simple chunking
        if not code_blocks or not self.preserve_code_blocks:
            return self._simple_chunk(text)
        
        # Replace code blocks with placeholders
        placeholder_text = text
        for i, block in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            placeholder_text = placeholder_text.replace(block, placeholder)
        
        # Chunk the text with placeholders
        chunks = self._simple_chunk(placeholder_text)
        
        # Restore code blocks in chunks
        restored_chunks = []
        for chunk in chunks:
            restored_chunk = chunk
            for i, block in enumerate(code_blocks):
                placeholder = f"__CODE_BLOCK_{i}__"
                if placeholder in restored_chunk:
                    restored_chunk = restored_chunk.replace(placeholder, block)
            restored_chunks.append(restored_chunk)
        
        return restored_chunks
    
    def _extract_code_blocks(self, text):
        """Extract code blocks from text"""
        # Match markdown code blocks
        code_blocks = re.findall(r'```[\w]*\n[\s\S]*?\n```', text)
        
        # Match indented code blocks
        indented_blocks = re.findall(r'(?:^|\n)( {4,}[^\n]+(?:\n {4,}[^\n]+)*)', text)
        code_blocks.extend(indented_blocks)
        
        return code_blocks
    
    def _simple_chunk(self, text):
        """Split text into chunks of specified size with overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            # Determine end position
            end = start + self.chunk_size
            
            # If we're at the end of the text, just add the remaining text
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Find a good breaking point (end of sentence, paragraph, etc.)
            break_point = self._find_break_point(text[start:end])
            
            # If no good breaking point found, just use the chunk size
            if break_point == 0:
                break_point = self.chunk_size
            
            # Add chunk
            chunks.append(text[start:start + break_point])
            
            # Move start position for next chunk, accounting for overlap
            start = start + break_point - self.chunk_overlap
            
            # Ensure we're making progress
            if start <= 0:
                start = break_point
        
        return chunks
    
    def _find_break_point(self, text):
        """Find a good breaking point in the text (end of sentence, paragraph)"""
        # Try to find paragraph break
        match = re.search(r'\n\s*\n', text)
        if match:
            return match.start()
        
        # Try to find sentence break
        matches = list(re.finditer(r'[.!?]\s+', text))
        if matches:
            # Find the last sentence break
            return matches[-1].end()
        
        # Try to find line break
        match = re.search(r'\n', text)
        if match:
            return match.start()
        
        return 0 