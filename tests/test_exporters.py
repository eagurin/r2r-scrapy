import os
import json
import pytest
import tempfile
from r2r_scrapy.exporters.r2r_exporter import R2RExporter
from r2r_scrapy.exporters.file_exporter import FileExporter

@pytest.mark.asyncio
async def test_r2r_exporter():
    exporter = R2RExporter(
        api_url='http://test.com',
        api_key='test_key',
        batch_size=2,
        max_concurrency=1
    )
    
    # Test initialization
    await exporter.initialize()
    assert exporter.session is not None
    
    # Test document export
    documents = [
        {
            'content': 'Test content 1',
            'metadata': {'title': 'Test 1'},
            'url': 'http://test.com/1'
        },
        {
            'content': 'Test content 2',
            'metadata': {'title': 'Test 2'},
            'url': 'http://test.com/2'
        }
    ]
    
    # Mock the API call (in real tests, you would use a proper mocking library)
    async def mock_post(*args, **kwargs):
        class MockResponse:
            async def json(self):
                return {'success': True}
            status = 200
        return MockResponse()
    
    exporter.session.post = mock_post
    
    result = await exporter.export_documents(documents, 'test_collection')
    assert len(result) == 2
    
    # Test cleanup
    await exporter.close()
    assert exporter.session is None

def test_file_exporter():
    with tempfile.TemporaryDirectory() as temp_dir:
        exporter = FileExporter(output_dir=temp_dir, format='json')
        
        # Test document export
        documents = [
            {
                'content': 'Test content 1',
                'metadata': {'title': 'Test 1'},
                'url': 'http://test.com/1',
                'document_id': 'doc1'
            },
            {
                'content': 'Test content 2',
                'metadata': {'title': 'Test 2'},
                'url': 'http://test.com/2',
                'document_id': 'doc2'
            }
        ]
        
        results = exporter.export_documents(documents, 'test_collection')
        assert len(results) == 2
        assert all(result['success'] for result in results)
        
        # Verify files were created
        docs_dir = os.path.join(temp_dir, 'docs', 'test_collection')
        assert os.path.exists(docs_dir)
        assert len(os.listdir(docs_dir)) == 2
        
        # Test metadata export
        metadata_dir = os.path.join(temp_dir, 'metadata')
        assert os.path.exists(metadata_dir)
        assert len(os.listdir(metadata_dir)) >= 2  # At least 2 metadata files
        
        # Test document content
        with open(os.path.join(docs_dir, 'doc1.json'), 'r') as f:
            doc1 = json.load(f)
            assert doc1['content'] == 'Test content 1'
            assert doc1['metadata']['title'] == 'Test 1'
        
        # Test chunks export
        chunks = ['Chunk 1', 'Chunk 2', 'Chunk 3']
        chunk_results = exporter.export_chunks(chunks, 'doc1')
        assert len(chunk_results) == 3
        assert all(result['success'] for result in chunk_results)
        
        chunks_dir = os.path.join(temp_dir, 'chunks', 'doc1')
        assert os.path.exists(chunks_dir)
        assert len(os.listdir(chunks_dir)) == 3 