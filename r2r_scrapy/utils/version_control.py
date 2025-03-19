import os
import re
import git
import logging
import hashlib
import json
from datetime import datetime

class VersionControl:
    """Track and manage document versions"""
    
    def __init__(self, storage_path='./versions'):
        self.logger = logging.getLogger(__name__)
        self.storage_path = storage_path
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize Git repository if it doesn't exist
        self.repo_path = os.path.join(storage_path, 'repo')
        os.makedirs(self.repo_path, exist_ok=True)
        
        try:
            self.repo = git.Repo(self.repo_path)
        except git.exc.InvalidGitRepositoryError:
            self.repo = git.Repo.init(self.repo_path)
            # Create initial commit
            open(os.path.join(self.repo_path, 'README.md'), 'w').write('# Document Version Control\n')
            self.repo.git.add('README.md')
            self.repo.git.commit('-m', 'Initial commit')
    
    def add_document(self, document, doc_id=None):
        """Add or update a document in version control"""
        # Generate document ID if not provided
        if not doc_id:
            doc_id = self._generate_document_id(document)
        
        # Create document path
        doc_path = os.path.join(self.repo_path, f"{doc_id}.json")
        
        # Check if document exists
        is_new = not os.path.exists(doc_path)
        
        # Calculate document hash
        doc_hash = self._calculate_document_hash(document)
        
        # Check if document has changed
        if not is_new:
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    existing_doc = json.load(f)
                existing_hash = existing_doc.get('hash')
                
                if existing_hash == doc_hash:
                    self.logger.debug(f"Document {doc_id} has not changed, skipping version control")
                    return {
                        'document_id': doc_id,
                        'changed': False,
                        'version': existing_doc.get('version', 1),
                    }
            except Exception as e:
                self.logger.error(f"Error reading existing document: {e}")
        
        # Prepare document data with version info
        version_data = {
            'document_id': doc_id,
            'content': document.get('content', ''),
            'metadata': document.get('metadata', {}),
            'url': document.get('url', ''),
            'title': document.get('title', ''),
            'hash': doc_hash,
            'version': 1 if is_new else (existing_doc.get('version', 1) + 1),
            'timestamp': datetime.now().isoformat(),
        }
        
        # Write document to file
        with open(doc_path, 'w', encoding='utf-8') as f:
            json.dump(version_data, f, ensure_ascii=False, indent=2)
        
        # Add to Git
        self.repo.git.add(doc_path)
        
        # Commit changes
        commit_message = f"{'Add' if is_new else 'Update'} document {doc_id} (version {version_data['version']})"
        self.repo.git.commit('-m', commit_message)
        
        return {
            'document_id': doc_id,
            'changed': True,
            'version': version_data['version'],
            'is_new': is_new,
        }
    
    def get_document_history(self, doc_id):
        """Get version history for a document"""
        doc_path = f"{doc_id}.json"
        
        try:
            # Get commit history for the document
            commits = list(self.repo.iter_commits(paths=doc_path))
            
            history = []
            for commit in commits:
                # Get document content at this commit
                try:
                    content = self.repo.git.show(f"{commit.hexsha}:{doc_path}")
                    doc_data = json.loads(content)
                    
                    history.append({
                        'commit_id': commit.hexsha,
                        'version': doc_data.get('version', 1),
                        'timestamp': commit.committed_datetime.isoformat(),
                        'message': commit.message,
                    })
                except Exception as e:
                    self.logger.error(f"Error getting document at commit {commit.hexsha}: {e}")
            
            return history
        except Exception as e:
            self.logger.error(f"Error getting document history: {e}")
            return []
    
    def get_document_version(self, doc_id, version=None, commit_id=None):
        """Get a specific version of a document"""
        doc_path = f"{doc_id}.json"
        
        try:
            if commit_id:
                # Get document at specific commit
                content = self.repo.git.show(f"{commit_id}:{doc_path}")
                return json.loads(content)
            elif version:
                # Find commit with specific version
                commits = list(self.repo.iter_commits(paths=doc_path))
                
                for commit in commits:
                    try:
                        content = self.repo.git.show(f"{commit.hexsha}:{doc_path}")
                        doc_data = json.loads(content)
                        
                        if doc_data.get('version') == version:
                            return doc_data
                    except Exception:
                        continue
                
                raise ValueError(f"Version {version} not found for document {doc_id}")
            else:
                # Get latest version
                with open(os.path.join(self.repo_path, doc_path), 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Error getting document version: {e}")
            return None
    
    def compare_versions(self, doc_id, version1, version2):
        """Compare two versions of a document"""
        doc1 = self.get_document_version(doc_id, version=version1)
        doc2 = self.get_document_version(doc_id, version=version2)
        
        if not doc1 or not doc2:
            return None
        
        # Compare content
        import difflib
        d = difflib.Differ()
        content1 = doc1.get('content', '').splitlines()
        content2 = doc2.get('content', '').splitlines()
        diff = list(d.compare(content1, content2))
        
        return {
            'document_id': doc_id,
            'version1': version1,
            'version2': version2,
            'timestamp1': doc1.get('timestamp'),
            'timestamp2': doc2.get('timestamp'),
            'diff': diff,
            'metadata_changed': doc1.get('metadata') != doc2.get('metadata'),
        }
    
    def _generate_document_id(self, document):
        """Generate a unique document ID"""
        url = document.get('url', '')
        title = document.get('title', '')
        content_sample = document.get('content', '')[:1000]  # Use first 1000 chars of content
        
        # Create a unique string
        unique_string = f"{url}:{title}:{content_sample}"
        
        # Generate MD5 hash
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def _calculate_document_hash(self, document):
        """Calculate a hash for document content"""
        content = document.get('content', '')
        metadata = json.dumps(document.get('metadata', {}), sort_keys=True)
        
        # Create a string to hash
        hash_string = f"{content}:{metadata}"
        
        # Generate MD5 hash
        return hashlib.md5(hash_string.encode()).hexdigest() 