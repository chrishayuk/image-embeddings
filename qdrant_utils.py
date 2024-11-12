# qdrant_utils.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

from hash_utils import hash_to_uuid

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

def find_similar_embeddings(query_embedding, top_k, with_payload = True):
    """Find similar embeddings to the query embedding"""

    # Initialize Qdrant client
    client = get_qdrant_client()
    
    # Search for similar embeddings
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        # Retrieve stored metadata (URLs)
        with_payload=with_payload  
    )

    # return the results
    return search_results
