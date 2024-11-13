from annoy import AnnoyIndex
import os
import json
import numpy as np
from tqdm import tqdm

# Constants
INDEX_FILE = "annoy_index.ann"
METADATA_FILE = "metadata.json"

def load_metadata():
    """
    Load metadata from the metadata file.
    """
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError("Metadata file does not exist.")
    with open(METADATA_FILE, "r") as f:
        return json.load(f)


def save_metadata(metadata):
    """
    Save metadata to the metadata file.
    """
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)


def rebuild_index(vector_size, metadata):
    """
    Rebuild an Annoy index in memory from the metadata.
    """
    index = AnnoyIndex(vector_size, metric="angular")
    for idx, payload in metadata.items():
        vector = np.array(payload["vector"])
        index.add_item(int(idx), vector)
    return index


def add_embeddings_to_index(index, metadata, embeddings, image_urls, start_idx=0):
    """
    Add embeddings and corresponding metadata to the Annoy index.
    """
    for idx, (embedding, url) in tqdm(
        enumerate(zip(embeddings, image_urls), start=start_idx),
        desc="Adding embeddings"
    ):
        normalized_embedding = embedding / np.linalg.norm(embedding)
        index.add_item(idx, normalized_embedding)
        metadata[str(idx)] = {"url": url, "vector": normalized_embedding.tolist()}
    
    # return the length of the metadata
    return len(metadata)


def create_collection_if_not_exists(vector_size):
    """
    Create the Annoy index and metadata file if they don't already exist.
    """
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        print("Collection does not exist. Creating new collection...")
        create_and_overwrite_collection(vector_size)
    else:
        print("Collection already exists.")


def create_and_overwrite_collection(vector_size, embeddings=None, image_urls=None):
    """
    Create a new Annoy index and metadata file, overwriting existing ones.
    Optionally, add initial embeddings to the index.
    """
    index = AnnoyIndex(vector_size, metric="angular")
    metadata = {}

    # check we have embeddings and images
    if embeddings is not None and image_urls is not None:
        print("Adding initial embeddings to the new collection...")
        add_embeddings_to_index(index, metadata, embeddings, image_urls)

    # build index and save
    index.build(10)
    index.save(INDEX_FILE)
    save_metadata(metadata)

    # added embeddings
    print(f"Collection created with {len(metadata)} embeddings: {INDEX_FILE}, {METADATA_FILE}")


def upload_embeddings(new_embeddings, new_image_urls):
    """
    Add new embeddings to the existing Annoy index.
    """
    if not os.path.exists(METADATA_FILE) or not os.path.exists(INDEX_FILE):
        raise FileNotFoundError("Collection does not exist. Create the collection first.")
    
    # load the metadata
    metadata = load_metadata()
    vector_size = len(new_embeddings[0])

    print("Rebuilding in-memory Annoy index...")
    index = rebuild_index(vector_size, metadata)

    print("Adding new embeddings...")
    start_idx = len(metadata)
    add_embeddings_to_index(index, metadata, new_embeddings, new_image_urls, start_idx)

    # build index and save
    index.build(10)
    index.save(INDEX_FILE)
    save_metadata(metadata)

    # added embeddings
    print(f"Added {len(new_embeddings)} new embeddings. Total: {len(metadata)}")


class SearchResult:
    """
    Custom class to mimic the expected structure for similarity results.
    """
    def __init__(self, score, payload):
        self.score = score
        self.payload = payload


def find_similar_embeddings(query_embedding, top_k, with_payload=True):
    """
    Find similar embeddings to the query embedding using cosine similarity.
    """
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        raise FileNotFoundError("Collection does not exist. Create the collection first.")

    metadata = load_metadata()
    vector_size = len(query_embedding)

    # Load the Annoy index
    index = AnnoyIndex(vector_size, metric="angular")
    index.load(INDEX_FILE)

    # Normalize the query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Perform nearest neighbor search
    nearest_ids, distances = index.get_nns_by_vector(query_embedding, top_k, include_distances=True)

    # Format results
    results = [
        SearchResult(1 - (dist / 2), metadata.get(str(idx), {}) if with_payload else {})
        for idx, dist in zip(nearest_ids, distances)
    ]

    return results
