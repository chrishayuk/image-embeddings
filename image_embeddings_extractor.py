# image_embeddings_extractor.py
from transformers import AutoModel
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings_model(model_name='jinaai/jina-clip-v1'):
    """Load the image embedding model."""
    return AutoModel.from_pretrained(model_name, trust_remote_code=True)

def extract_embeddings(model, image_urls):
    """Extract embeddings for a list of image URLs."""
    embeddings = model.encode_image(image_urls)

    # Convert to JSON-compatible format
    return np.array(embeddings)

def calculate_cosine_similarity(embedding1, embedding2):
    """Calculate the cosine similarity between two embeddings."""
    return cosine_similarity([embedding1], [embedding2])[0][0]

if __name__ == "__main__":
    # Example usage of the embedding extraction process
    sample_image_urls = [
        "https://i.pinimg.com/600x315/21/48/7e/21487e8e0970dd366dafaed6ab25d8d8.jpg",
        "https://i.pinimg.com/736x/c9/f2/3e/c9f23e212529f13f19bad5602d84b78b.jpg",
    ]

    # Load the embeddings model
    print("Loading model...")
    model = load_embeddings_model()

    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings(model, sample_image_urls)

    # Display the embeddings in JSON format
    print("Embeddings (JSON-compatible):")
    print(json.dumps(embeddings.tolist(), indent=2))

    # Calculate cosine similarity between the two sample images
    if len(embeddings) >= 2:
        cosine_sim = calculate_cosine_similarity(embeddings[0], embeddings[1])
        print(f"Cosine similarity between the images: {cosine_sim:.4f}")
    else:
        print("Not enough images to calculate cosine similarity.")
