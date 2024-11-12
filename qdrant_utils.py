# qdrant_utils.py
import hashlib
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

# Constants
COLLECTION_NAME = "images"

def get_qdrant_client():
    """Initialize and return the Qdrant client."""
    return QdrantClient(url="http://localhost:6333")

def create_collection_if_not_exists(vector_size):
    """Create the Qdrant collection if it doesn't already exist."""
    client = get_qdrant_client()
    
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

def create_and_overwrite_collection(vector_size):
    """Create the Qdrant collection, overwriting if it already exists."""
    # get the client
    client = get_qdrant_client()

    # delete the collection if already exists
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    # create a brand new collection
    client.create_collection(
        COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

def hash_to_uuid(embedding):
    """Generate a UUID based on an SHA-256 hash of the embedding."""
    embedding_str = ','.join(map(str, embedding))  # Convert to a comma-separated string
    hash_value = hashlib.sha256(embedding_str.encode()).hexdigest()
    return str(uuid.UUID(hash_value[:32]))  # Use first 32 characters to create a UUID

def upload_embeddings(embeddings, image_urls):
    """Upload embeddings to the Qdrant collection with a progress bar, UUID-based IDs, and image URLs."""
    client = get_qdrant_client()

    # Create points with UUIDs and store the image URL in the payload
    points = [
        PointStruct(id=hash_to_uuid(embedding), vector=embedding, payload={"url": url})
        for embedding, url in zip(embeddings, image_urls)
    ]

    # Use tqdm to show progress
    for point in tqdm(points, desc="Uploading embeddings to Qdrant"):
        client.upsert(COLLECTION_NAME, points=[point])