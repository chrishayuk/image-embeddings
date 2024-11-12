# image_embeddings_extractor.py
from transformers import AutoModel
import numpy as np

def load_embeddings_model(model_name='jinaai/jina-clip-v1'):
    """Load the image embedding model."""
    return AutoModel.from_pretrained(model_name, trust_remote_code=True)

def extract_embeddings(model, image_urls):
    """Extract embeddings for a list of image URLs."""
    embeddings = model.encode_image(image_urls)

    # Convert to JSON-compatible format
    return np.array(embeddings).tolist()
