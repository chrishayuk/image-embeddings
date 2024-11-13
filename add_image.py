import argparse
import vector_utils
from image_embeddings_extractor import load_embeddings_model, extract_embeddings


def add_image(model, image_url):
    """Extract embedding for a single image and upload it to the selected backend."""
    # Extract embedding for the single image URL
    embeddings = extract_embeddings(model, [image_url])
    
    # Upload the single embedding to the backend
    vector_utils.backend.upload_embeddings(embeddings, [image_url])


if __name__ == "__main__":
    # Setup the parser
    parser = argparse.ArgumentParser(description="Add an image embedding to the vector database.")

    # Arguments
    parser.add_argument("--image_url", type=str, required=True, help="URL of the image to add")
    parser.add_argument("--vector_size", type=int, default=768, help="Vector size of the image embedding")
    parser.add_argument("--backend", type=str, default="annoy", choices=["annoy", "qdrant"], help="Vector backend to use")

    # Parse arguments
    args = parser.parse_args()

    # Dynamically load the utility functions
    vector_utils.load_vector_utils(args.backend)

    # Load the model
    model = load_embeddings_model()

    # Ensure collection exists
    vector_utils.backend.create_collection_if_not_exists(args.vector_size)

    # Process and upload the image
    add_image(model, args.image_url)

    # Show the output
    print(f"Embedding for image '{args.image_url}' has been uploaded to the {args.backend} backend.")
