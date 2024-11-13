import argparse
from image_embeddings_extractor import load_embeddings_model, extract_embeddings
#from qdrant_utils import find_similar_embeddings
from annoy_utils import find_similar_embeddings
from display_utils import display_similarity_results

def find_similar_images(model, image_url, top_k=5):
    """Find similar images to the given image URL."""
    # Extract embedding for the query image
    query_embedding = extract_embeddings(model, [image_url])[0]
    
    # Search for similar embeddings
    search_results = find_similar_embeddings(query_embedding, top_k)
    
    # Display results
    display_similarity_results(search_results, image_url, top_k)

if __name__ == "__main__":
    # Setup the parser
    parser = argparse.ArgumentParser(description="Find similar images in Qdrant.")

    # Arguments
    parser.add_argument("--image_url", type=str, help="URL of the image to find similar images for")
    parser.add_argument("--top_k", type=int, default=5, help="Number of similar images to retrieve")

    # Parse arguments
    args = parser.parse_args()

    # Load the model
    model = load_embeddings_model()
    
    # Find similar images
    find_similar_images(model, args.image_url, args.top_k)
