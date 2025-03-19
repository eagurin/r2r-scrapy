import asyncio
import aiohttp
import json
import hashlib
import uuid
from datetime import datetime

class R2RExporter:
    """Export processed documents to R2R API"""
    
    def __init__(self, api_url, api_key, batch_size=10, max_concurrency=5):
        self.api_url = api_url
        self.api_key = api_key
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.session = None
    
    async def initialize(self):
        """Initialize the exporter"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close the exporter"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def export_documents(self, documents, collection_id=None):
        """Export documents to R2R API"""
        await self.initialize()
        
        # Process documents in batches
        results = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            batch_results = await self.process_batch(batch, collection_id)
            results.extend(batch_results)
        
        return results
    
    async def process_batch(self, batch, collection_id=None):
        """Process a batch of documents"""
        async with self.semaphore:
            try:
                # Prepare data for R2R API
                r2r_data = self.prepare_r2r_data(batch, collection_id)
                
                # Send data to R2R API
                async with self.session.post(
                    f"{self.api_url}/documents",
                    json=r2r_data,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"R2R API error: {response.status} - {error_text}")
                    
                    result = await response.json()
                    return result
            except Exception as e:
                # Handle errors with retry logic
                print(f"Error sending data to R2R: {e}")
                # Implement retry logic here
                return [{"error": str(e), "document": doc} for doc in batch]
    
    def prepare_r2r_data(self, batch, collection_id=None):
        """Prepare data in the format expected by R2R API"""
        documents = []
        
        for item in batch:
            # Generate document ID if not provided
            doc_id = item.get('document_id') or self.generate_document_id(item)
            
            # Determine collection ID
            coll_id = collection_id or self.get_collection_id(item)
            
            # Determine chunking strategy based on content type
            chunk_strategy = self.determine_chunk_strategy(item)
            
            # Create document object
            doc = {
                "content": item['content'],
                "metadata": item.get('metadata', {}),
                "document_id": doc_id,
                "collection_id": coll_id,
                "chunk_strategy": chunk_strategy
            }
            documents.append(doc)
        
        return {"documents": documents}
    
    def generate_document_id(self, item):
        """Generate a unique document ID"""
        url = item.get('url', '')
        content_hash = hashlib.md5(item.get('content', '').encode()).hexdigest()
        timestamp = datetime.now().isoformat()
        
        # Create a unique ID based on URL, content hash, and timestamp
        unique_string = f"{url}:{content_hash}:{timestamp}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, unique_string))
    
    def get_collection_id(self, item):
        """Determine collection ID based on metadata"""
        metadata = item.get('metadata', {})
        library_name = metadata.get('library_name', '')
        version = metadata.get('version', 'latest')
        
        # Create a collection ID based on library name and version
        if library_name:
            return f"{library_name.lower().replace(' ', '_')}_{version}"
        
        # Fallback to a default collection
        return "documentation_collection"
    
    def determine_chunk_strategy(self, item):
        """Determine the best chunking strategy based on content type"""
        metadata = item.get('metadata', {})
        doc_type = metadata.get('doc_type', '')
        
        if doc_type == 'api_reference':
            return "semantic"
        elif doc_type == 'tutorial':
            return "markdown_header"
        elif doc_type == 'code_example':
            return "code_aware"
        
        # Default to semantic chunking
        return "semantic"
    
    async def create_collection(self, collection_id, metadata=None):
        """Create a new collection in R2R"""
        await self.initialize()
        
        try:
            async with self.session.post(
                f"{self.api_url}/collections",
                json={
                    "collection_id": collection_id,
                    "metadata": metadata or {}
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status not in (200, 201):
                    error_text = await response.text()
                    raise Exception(f"R2R API error: {response.status} - {error_text}")
                
                result = await response.json()
                return result
        except Exception as e:
            print(f"Error creating collection: {e}")
            return {"error": str(e)} 