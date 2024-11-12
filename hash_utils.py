import hashlib
import uuid

def hash_to_uuid(embedding):
    """Generate a UUID based on an SHA-256 hash of the embedding."""
    # get a string version of the embedding
    embedding_str = ','.join(map(str, embedding))  
    
    # Convert to a comma-separated string
    hash_value = hashlib.sha256(embedding_str.encode()).hexdigest()

    # Use first 32 characters to create a UUID
    return str(uuid.UUID(hash_value[:32]))  