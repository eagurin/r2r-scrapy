import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticChunker:
    """Split text into semantically coherent chunks"""
    
    def __init__(self, chunk_size=800, chunk_overlap=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def chunk_text(self, text):
        """Split text into semantically coherent chunks"""
        if not text:
            return []
        
        # First, split text into paragraphs
        paragraphs = self._split_into_paragraphs(text)
        
        # If text is short enough, return as a single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        # Create initial chunks based on size
        initial_chunks = self._create_initial_chunks(paragraphs)
        
        # Refine chunks based on semantic coherence
        refined_chunks = self._refine_chunks(initial_chunks)
        
        return refined_chunks
    
    def _split_into_paragraphs(self, text):
        """Split text into paragraphs"""
        # Split by double newlines (common paragraph separator)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def _create_initial_chunks(self, paragraphs):
        """Create initial chunks based on size constraints"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph_size = len(paragraph)
            
            # If adding this paragraph exceeds chunk size and we already have content,
            # finish the current chunk and start a new one
            if current_size + paragraph_size > self.chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_size = 0
                overlap_paragraphs = []
                
                # Add paragraphs from the end of the previous chunk for overlap
                for p in reversed(current_chunk):
                    if overlap_size + len(p) <= self.chunk_overlap:
                        overlap_paragraphs.insert(0, p)
                        overlap_size += len(p)
                    else:
                        break
                
                current_chunk = overlap_paragraphs
                current_size = overlap_size
            
            # Add paragraph to current chunk
            current_chunk.append(paragraph)
            current_size += paragraph_size
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _refine_chunks(self, chunks):
        """Refine chunks based on semantic coherence"""
        if len(chunks) <= 1:
            return chunks
        
        # Calculate TF-IDF vectors for chunks
        try:
            tfidf_matrix = self.vectorizer.fit_transform(chunks)
            
            # Calculate similarity between adjacent chunks
            similarities = []
            for i in range(len(chunks) - 1):
                similarity = cosine_similarity(
                    tfidf_matrix[i:i+1], tfidf_matrix[i+1:i+2]
                )[0][0]
                similarities.append(similarity)
            
            # Identify low-similarity boundaries (potential topic changes)
            mean_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            threshold = max(0.1, mean_similarity - std_similarity)
            
            # Merge chunks with high similarity
            refined_chunks = []
            current_chunk = chunks[0]
            
            for i, similarity in enumerate(similarities):
                if similarity < threshold:
                    # Low similarity indicates a topic change, keep chunks separate
                    refined_chunks.append(current_chunk)
                    current_chunk = chunks[i + 1]
                else:
                    # High similarity, merge chunks with proper overlap handling
                    # Find a good breaking point (end of sentence, paragraph, etc.)
                    break_point = self._find_break_point(chunks[i + 1])
                    if break_point > 0:
                        current_chunk += '\n\n' + chunks[i + 1][:break_point]
                        refined_chunks.append(current_chunk)
                        current_chunk = chunks[i + 1][break_point:]
                    else:
                        current_chunk += '\n\n' + chunks[i + 1]
            
            # Add the last chunk
            if current_chunk:
                refined_chunks.append(current_chunk)
            
            return refined_chunks
        except Exception:
            # Fallback to original chunks if TF-IDF processing fails
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
            # Find a sentence break near the middle
            middle = len(text) // 2
            closest_match = min(matches, key=lambda m: abs(m.end() - middle))
            return closest_match.end()
        
        return 0 