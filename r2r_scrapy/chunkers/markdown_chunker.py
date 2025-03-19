import re

class MarkdownChunker:
    """Split markdown text into chunks based on headings and size"""
    
    def __init__(self, chunk_size=800, chunk_overlap=150, heading_split=True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.heading_split = heading_split
    
    def chunk_text(self, text):
        """Split markdown text into chunks based on headings and size"""
        if not text:
            return []
        
        # If text is short enough, return as a single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        # If heading split is enabled, split by headings first
        if self.heading_split:
            heading_chunks = self._split_by_headings(text)
            
            # Further split large chunks by size
            final_chunks = []
            for chunk in heading_chunks:
                if len(chunk) <= self.chunk_size:
                    final_chunks.append(chunk)
                else:
                    final_chunks.extend(self._split_by_size(chunk))
            
            return final_chunks
        else:
            # Split directly by size
            return self._split_by_size(text)
    
    def _split_by_headings(self, text):
        """Split text by markdown headings"""
        # Match all headings (# to ######)
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+?)(?:\s+#{1,6})?$', re.MULTILINE)
        
        # Find all headings
        headings = list(heading_pattern.finditer(text))
        
        # If no headings found, return the whole text
        if not headings:
            return [text]
        
        # Split text by headings
        chunks = []
        for i, heading in enumerate(headings):
            start = heading.start()
            
            # Determine end position
            if i < len(headings) - 1:
                end = headings[i + 1].start()
            else:
                end = len(text)
            
            # Add chunk
            chunks.append(text[start:end])
        
        # Add text before the first heading if it exists
        if headings[0].start() > 0:
            chunks.insert(0, text[:headings[0].start()])
        
        return chunks
    
    def _split_by_size(self, text):
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
            
            # Find a good breaking point (end of paragraph, sentence, etc.)
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
        """Find a good breaking point in markdown text"""
        # Try to find paragraph break
        match = re.search(r'\n\s*\n', text)
        if match:
            return match.start()
        
        # Try to find heading
        match = re.search(r'\n#{1,6}\s+', text)
        if match:
            return match.start()
        
        # Try to find list item
        match = re.search(r'\n[*-]\s+', text)
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