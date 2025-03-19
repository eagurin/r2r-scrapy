import os
import json
import yaml
import hashlib
from datetime import datetime

class FileExporter:
    """Export documents to local files"""
    
    def __init__(self, output_dir='./output', format='json'):
        self.output_dir = output_dir
        self.format = format.lower()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for different content types
        self.docs_dir = os.path.join(output_dir, 'docs')
        self.chunks_dir = os.path.join(output_dir, 'chunks')
        self.metadata_dir = os.path.join(output_dir, 'metadata')
        
        os.makedirs(self.docs_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
    
    def export_documents(self, documents, collection_id=None):
        """Export documents to files"""
        results = []
        
        for doc in documents:
            # Generate document ID if not provided
            doc_id = doc.get('document_id') or self._generate_document_id(doc)
            
            # Determine collection directory
            collection_dir = os.path.join(self.docs_dir, collection_id or 'default')
            os.makedirs(collection_dir, exist_ok=True)
            
            # Export document
            result = self._export_document(doc, doc_id, collection_dir)
            results.append(result)
        
        # Export collection metadata
        if collection_id:
            self._export_collection_metadata(collection_id, len(results))
        
        return results
    
    def _export_document(self, document, doc_id, collection_dir):
        """Export a single document"""
        # Prepare document data
        doc_data = {
            'document_id': doc_id,
            'content': document.get('content', ''),
            'metadata': document.get('metadata', {}),
            'url': document.get('url', ''),
            'title': document.get('title', ''),
            'export_timestamp': datetime.now().isoformat(),
        }
        
        # Determine file path
        file_name = f"{doc_id}.{self.format}"
        file_path = os.path.join(collection_dir, file_name)
        
        # Write document to file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if self.format == 'json':
                    json.dump(doc_data, f, ensure_ascii=False, indent=2)
                elif self.format == 'yaml':
                    yaml.dump(doc_data, f, allow_unicode=True)
                else:  # Plain text
                    f.write(doc_data['content'])
            
            # Export metadata separately
            self._export_metadata(doc_data, doc_id)
            
            return {
                'document_id': doc_id,
                'file_path': file_path,
                'success': True,
            }
        except Exception as e:
            return {
                'document_id': doc_id,
                'error': str(e),
                'success': False,
            }
    
    def _export_metadata(self, doc_data, doc_id):
        """Export document metadata separately"""
        metadata = {
            'document_id': doc_id,
            'metadata': doc_data['metadata'],
            'url': doc_data.get('url', ''),
            'title': doc_data.get('title', ''),
            'content_length': len(doc_data.get('content', '')),
            'export_timestamp': doc_data.get('export_timestamp'),
        }
        
        # Write metadata to file
        metadata_path = os.path.join(self.metadata_dir, f"{doc_id}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def _export_collection_metadata(self, collection_id, doc_count):
        """Export collection metadata"""
        metadata = {
            'collection_id': collection_id,
            'document_count': doc_count,
            'export_timestamp': datetime.now().isoformat(),
        }
        
        # Write metadata to file
        metadata_path = os.path.join(self.metadata_dir, f"{collection_id}_collection.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def _generate_document_id(self, document):
        """Generate a unique document ID"""
        url = document.get('url', '')
        title = document.get('title', '')
        content_sample = document.get('content', '')[:1000]  # Use first 1000 chars of content
        timestamp = datetime.now().isoformat()
        
        # Create a unique string
        unique_string = f"{url}:{title}:{content_sample}:{timestamp}"
        
        # Generate MD5 hash
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def export_chunks(self, chunks, doc_id):
        """Export chunks to files"""
        results = []
        
        # Create directory for document chunks
        chunks_doc_dir = os.path.join(self.chunks_dir, doc_id)
        os.makedirs(chunks_doc_dir, exist_ok=True)
        
        for i, chunk in enumerate(chunks):
            # Prepare chunk data
            chunk_data = {
                'document_id': doc_id,
                'chunk_id': f"{doc_id}_{i}",
                'content': chunk,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'export_timestamp': datetime.now().isoformat(),
            }
            
            # Determine file path
            file_name = f"chunk_{i}.{self.format}"
            file_path = os.path.join(chunks_doc_dir, file_name)
            
            # Write chunk to file
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    if self.format == 'json':
                        json.dump(chunk_data, f, ensure_ascii=False, indent=2)
                    elif self.format == 'yaml':
                        yaml.dump(chunk_data, f, allow_unicode=True)
                    else:  # Plain text
                        f.write(chunk_data['content'])
                
                results.append({
                    'chunk_id': chunk_data['chunk_id'],
                    'file_path': file_path,
                    'success': True,
                })
            except Exception as e:
                results.append({
                    'chunk_id': chunk_data['chunk_id'],
                    'error': str(e),
                    'success': False,
                })
        
        return results 