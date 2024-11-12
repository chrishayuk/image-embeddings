import argparse
from tabulate import tabulate
import pyshorteners
from image_embeddings_extractor import load_embeddings_model, extract_embeddings
from qdrant_utils import find_similar_embeddings

def shorten_url(url):
    """Shorten a URL using pyshorteners."""
    shortener = pyshorteners.Shortener()
    try:
        # shorten url
        return shortener.tinyurl.short(url)
    except Exception:
        # Return original URL if shortening fails
        return url  

def find_similar_images(model, image_url, top_k=5):
    """Find similar images to the given image URL."""
    # Extract embedding for the query image
    query_embedding = extract_embeddings(model, [image_url])[0]
    
    # Search for similar embeddings
    search_results = find_similar_embeddings(query_embedding, top_k)
    
    # Prepare data with shortened URLs
    table_data = []
    for result in search_results:
        similarity_score = f"{result.score:.4f}"  # Limit similarity score to 4 decimal places
        matched_url = result.payload.get("url", "No URL found")
        shortened_url = shorten_url(matched_url)
        table_data.append([shortened_url, similarity_score])
    
    # Display results in a tabulated format
    print(f"Top {top_k} similar images to '{image_url}':")
    print(tabulate(table_data, headers=["URL", "Similarity Score"], tablefmt="fancy_grid"))

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
