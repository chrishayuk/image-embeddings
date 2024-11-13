# create_index.py
import argparse
import json
from image_embeddings_extractor import load_embeddings_model, extract_embeddings
#from qdrant_utils import create_and_overwrite_collection, upload_embeddings
from annoy_utils import create_and_overwrite_collection, upload_embeddings

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

if __name__ == "__main__":
    # Setup the parser
    parser = argparse.ArgumentParser(description="Add image embeddings to Qdrant.")

    # Arguments
    parser.add_argument("--image_json", type=str, default='image_urls.json', help="Path to the JSON file containing image URLs")
    parser.add_argument("--vector_size", type=int, default=768, help="Vector size of the image embeddings")

    # Parse arguments
    args = parser.parse_args()

    # Load image URLs
    image_urls = load_image_urls(args.image_json)
    
    # Initialize the model
    model = load_embeddings_model()

    # Create the collection and overwrite if it exists
    create_and_overwrite_collection(args.vector_size)

    # Process and upload images
    process_and_upload_images(model, image_urls)
