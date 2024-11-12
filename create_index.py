# create_index.py
import json
from image_embeddings_extractor import load_embeddings_model, extract_embeddings
from qdrant_utils import create_and_overwrite_collection, upload_embeddings

def load_image_urls(file_path):
    """Load image URLs from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def process_and_upload_images(model, image_urls):
    """Extract embeddings for the given images and upload them to Qdrant."""
    # Extract the embeddings
    embeddings = extract_embeddings(model, image_urls)
    
    # Upload the embeddings to Qdrant with progress shown
    upload_embeddings(embeddings, image_urls)

def main(image_urls_file_path):
    # Load image URLs
    image_urls = load_image_urls(image_urls_file_path)
    
    # Initialize the model
    model = load_embeddings_model()

    # Get vector size from one embedding
    vector_size = len(extract_embeddings(model, [image_urls[0]])[0])  

    # Create the collection and overwrite if exists
    create_and_overwrite_collection(vector_size)
    
    # Process and upload images
    process_and_upload_images(model, image_urls)

if __name__ == "__main__":
    # Path to the JSON file with image URLs
    image_urls_file_path = 'image_urls.json'
    main(image_urls_file_path)
