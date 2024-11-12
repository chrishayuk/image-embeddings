# display_utils.py
import pyshorteners
from tabulate import tabulate

def shorten_url(url):
    """Shorten a URL using pyshorteners."""
    shortener = pyshorteners.Shortener()
    try:
        return shortener.tinyurl.short(url)
    except Exception:
        return url  # Return original URL if shortening fails

def display_similarity_results(search_results, image_url, top_k):
    """Prepare and display similarity results in a table with shortened URLs."""
    # Prepare data with shortened URLs
    table_data = []

    # loop through search results
    for result in search_results:
        similarity_score = f"{result.score:.4f}"  # Limit similarity score to 4 decimal places
        matched_url = result.payload.get("url", "No URL found")
        shortened_url = shorten_url(matched_url)
        table_data.append([shortened_url, similarity_score])

    # Display results in a tabulated format
    print(f"Top {top_k} similar images to '{image_url}':")
    print(tabulate(table_data, headers=["URL", "Similarity Score"], tablefmt="fancy_grid"))
