import argparse
from image_embeddings_extractor import load_embeddings_model, extract_embeddings
from qdrant_utils import get_qdrant_client, COLLECTION_NAME

def find_similar_images(image_url, top_k=5):
    """Find similar images to the given image URL."""
    # Load the model
    model = load_embeddings_model()
    
    # Extract embedding for the query image
    query_embedding = extract_embeddings(model, [image_url])[0]
    
    # Initialize Qdrant client
    client = get_qdrant_client()
    
    # Search for similar embeddings
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True  # Retrieve stored metadata (URLs)
    )
    
    # Print the results
    print(f"Top {top_k} similar images to '{image_url}':")
    for result in search_results:
        similarity_score = result.score
        matched_url = result.payload.get("url", "No URL found")
        print(f"URL: {matched_url} , Similarity Score: {similarity_score}")

def main():
    parser = argparse.ArgumentParser(description="Find similar images in Qdrant.")
    parser.add_argument("--image_url", type=str, help="URL of the image to find similar images for")
    parser.add_argument("--top_k", type=int, default=5, help="Number of similar images to retrieve")
    args = parser.parse_args()
    
    # Find similar images
    find_similar_images(args.image_url, args.top_k)

if __name__ == "__main__":
    main()
