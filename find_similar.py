import argparse
import vector_utils  # Import vector_utils to dynamically handle the backend
from image_embeddings_extractor import load_embeddings_model, extract_embeddings
from display_utils import display_similarity_results


def find_similar_images(model, image_url, top_k=5):
    """Find similar images to the given image URL."""
    # Extract embedding for the query image
    query_embedding = extract_embeddings(model, [image_url])[0]
    
    # Search for similar embeddings using the dynamically loaded backend
    search_results = vector_utils.backend.find_similar_embeddings(query_embedding, top_k)
    
    # Display results
    display_similarity_results(search_results, image_url, top_k)


if __name__ == "__main__":
    # Setup the parser
    parser = argparse.ArgumentParser(description="Find similar images in the vector database.")

    # Arguments
    parser.add_argument("--image_url", type=str, required=True, help="URL of the image to find similar images for")
    parser.add_argument("--top_k", type=int, default=5, help="Number of similar images to retrieve")
    parser.add_argument("--backend", type=str, default="annoy", choices=["annoy", "qdrant"], help="Vector backend to use")

    # Parse arguments
    args = parser.parse_args()

    # Dynamically load the utility functions for the selected backend
    vector_utils.load_vector_utils(args.backend)

    # Load the model
    model = load_embeddings_model()
    
    # Find and display similar images
    find_similar_images(model, args.image_url, args.top_k)
