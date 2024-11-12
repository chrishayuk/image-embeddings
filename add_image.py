# add_image.py
import argparse
from image_embeddings_extractor import load_embeddings_model, extract_embeddings
from qdrant_utils import get_qdrant_client, create_collection_if_not_exists, upload_embeddings

def add_image_to_qdrant(image_url):
    """Extract embedding for a single image and upload it to Qdrant."""
    # Load the model
    model = load_embeddings_model()
    
    # Extract embedding for the single image URL
    embeddings = extract_embeddings(model, [image_url])
    
    # Ensure collection exists (assuming embedding size can be derived from the embedding)
    vector_size = len(embeddings[0])
    create_collection_if_not_exists(vector_size)
    
    # Upload the single embedding to Qdrant
    upload_embeddings(embeddings, [image_url])

def main():
    # setup the parser
    parser = argparse.ArgumentParser(description="Add an image embedding to Qdrant.")

    # arguements
    parser.add_argument("--image_url", type=str, help="URL of the image to add to Qdrant")

    # parse
    args = parser.parse_args()
    
    # Process and upload the image
    add_image_to_qdrant(args.image_url)
    print(f"Embedding for image '{args.image_url}' has been uploaded to Qdrant.")

if __name__ == "__main__":
    main()
