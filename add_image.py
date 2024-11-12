# add_image.py
import argparse
from image_embeddings_extractor import load_embeddings_model, extract_embeddings
from qdrant_utils import create_collection_if_not_exists, upload_embeddings

def add_image(model, image_url):
    """Extract embedding for a single image and upload it to Qdrant."""
    # Extract embedding for the single image URL
    embeddings = extract_embeddings(model, [image_url])
    
    # Upload the single embedding to Qdrant
    upload_embeddings(embeddings, [image_url])

if __name__ == "__main__":
    # setup the parser
    parser = argparse.ArgumentParser(description="Add an image embedding to Qdrant.")

    # arguements
    parser.add_argument("--image_url", type=str, help="URL of the image to add to Qdrant")
    parser.add_argument("--vector_size", type=int, default=768, help="Vector size of the image embedding")

    # parse
    args = parser.parse_args()

    # Load the model
    model = load_embeddings_model()

    # Ensure collection exists (assuming embedding size can be derived from the embedding)
    vector_size = args.vector_size

    # create the collection if it doesn't exist
    create_collection_if_not_exists(vector_size)

    # Process and upload the image
    add_image(model, args.image_url)

    # show the output
    print(f"Embedding for image '{args.image_url}' has been uploaded to Qdrant.")
