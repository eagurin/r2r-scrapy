import re

class RecursiveChunker:
    def __init__(self, chunk_size=800, chunk_overlap=150, max_depth=3):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_depth = max_depth
    
    def chunk_text(self, text):
        """Split text into chunks using recursive approach"""
        if not text:
            return []
        
        # If text is short enough, return as a single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        # Start recursive chunking
        return self._recursive_chunk(text, 0)
    
    def _recursive_chunk(self, text, depth):
        """Recursively split text into chunks"""
        # If text is short enough or we've reached max depth, use simple chunking
        if len(text) <= self.chunk_size or depth >= self.max_depth:
            return self._simple_chunk(text)
        
        # Try to split by different delimiters based on depth
        if depth == 0:
            # First level: try to split by double newlines (paragraphs)
            delimiter = r'\n\s*\n'
        elif depth == 1:
            # Second level: try to split by headings or single newlines
            delimiter = r'\n#{1,6}\s+|\n'
        else:
            # Third level: try to split by sentences
            delimiter = r'[.!?]\s+'
        
        # Split text by delimiter
        parts = re.split(delimiter, text)
        
        # If splitting produced only one part, move to next depth
        if len(parts) <= 1:
            return self._recursive_chunk(text, depth + 1)
        
        # Recombine parts that are too small
        combined_parts = []
        current_part = ""
        
        for part in parts:
            # Skip empty parts
            if not part.strip():
                continue
            
            # If adding this part would exceed chunk size, finalize current part
            if len(current_part) + len(part) > self.chunk_size and current_part:
                combined_parts.append(current_part)
                current_part = part
            else:
                # Add delimiter back if not the first part
                if current_part:
                    if depth == 0:
                        current_part += "\n\n"
                    elif depth == 1:
                        current_part += "\n"
                    else:
                        current_part += ". "
                
                current_part += part
        
        # Add the last part if it exists
        if current_part:
            combined_parts.append(current_part)
        
        # If recombining didn't help, move to next depth
        if len(combined_parts) <= 1:
            return self._recursive_chunk(text, depth + 1)
        
        # Process each combined part recursively
        chunks = []
        for part in combined_parts:
            if len(part) <= self.chunk_size:
                chunks.append(part)
            else:
                chunks.extend(self._recursive_chunk(part, depth + 1))
        
        return chunks
    
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
        """Find a good breaking point in the text"""
        # Try to find paragraph break
        match = re.search(r'\n\s*\n', text)
        if match:
            return match.start()
        
        # Try to find heading
        match = re.search(r'\n#{1,6}\s+', text)
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