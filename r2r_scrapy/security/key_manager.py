import os
import json
import base64
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class KeyManager:
    """Secure management of API keys and sensitive data"""
    
    def __init__(self, storage_path=None, master_password=None):
        self.logger = logging.getLogger(__name__)
        
        # Storage path for keys
        self.storage_path = storage_path or os.path.join(os.path.expanduser('~'), '.r2r_scrapy', 'keys.json')
        
        # Create storage directory if it doesn't exist
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        # Master password for encryption
        self.master_password = master_password or os.environ.get('R2R_SCRAPY_MASTER_PASSWORD')
        
        # Initialize encryption key
        self.encryption_key = self._derive_key()
        
        # Load keys
        self.keys = self._load_keys()
    
    def _derive_key(self):
        """Derive encryption key from master password"""
        if not self.master_password:
            # If no master password, generate a random one and store it
            env_key_path = os.path.join(os.path.dirname(self.storage_path), '.env_key')
            
            if os.path.exists(env_key_path):
                # Load existing key
                with open(env_key_path, 'rb') as f:
                    key = f.read()
            else:
                # Generate new key
                key = Fernet.generate_key()
                
                # Save key
                with open(env_key_path, 'wb') as f:
                    f.write(key)
                
                # Set permissions
                os.chmod(env_key_path, 0o600)
            
            return key
        
        # Derive key from master password
        password = self.master_password.encode()
        salt = b'r2r_scrapy_salt'  # Fixed salt, could be improved
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def _load_keys(self):
        """Load keys from storage"""
        if not os.path.exists(self.storage_path):
            return {}
        
        try:
            with open(self.storage_path, 'r') as f:
                encrypted_data = json.load(f)
            
            # Decrypt data
            cipher = Fernet(self.encryption_key)
            decrypted_data = cipher.decrypt(encrypted_data['data'].encode()).decode()
            
            return json.loads(decrypted_data)
        except Exception as e:
            self.logger.error(f"Error loading keys: {e}")
            return {}
    
    def _save_keys(self):
        """Save keys to storage"""
        try:
            # Encrypt data
            cipher = Fernet(self.encryption_key)
            encrypted_data = cipher.encrypt(json.dumps(self.keys).encode()).decode()
            
            # Save to file
            with open(self.storage_path, 'w') as f:
                json.dump({'data': encrypted_data}, f)
            
            # Set permissions
            os.chmod(self.storage_path, 0o600)
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving keys: {e}")
            return False
    
    def get_key(self, key_name):
        """Get a key by name"""
        return self.keys.get(key_name)
    
    def set_key(self, key_name, key_value):
        """Set a key"""
        self.keys[key_name] = key_value
        return self._save_keys()
    
    def delete_key(self, key_name):
        """Delete a key"""
        if key_name in self.keys:
            del self.keys[key_name]
            return self._save_keys()
        return False
    
    def list_keys(self):
        """List all key names"""
        return list(self.keys.keys())
    
    def rotate_master_key(self, new_master_password):
        """Rotate the master encryption key"""
        # Save current keys
        current_keys = self.keys
        
        # Update master password
        self.master_password = new_master_password
        
        # Derive new key
        self.encryption_key = self._derive_key()
        
        # Set keys and save with new encryption
        self.keys = current_keys
        return self._save_keys() 