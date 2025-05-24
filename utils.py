import json
import re
from bs4 import BeautifulSoup
import mimetypes
import os

def preprocess_html(content):
    """Extract text from HTML content, removing tags."""
    soup = BeautifulSoup(content, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

def preprocess_text(content):
    """Return plain text content as-is."""
    return content

def escape_json_string(content):
    """
    Properly escape special characters for JSON compatibility.
    Uses json.dumps to handle all edge cases.
    """
    # Encode content as a JSON string and remove surrounding quotes
    escaped = json.dumps(content)[1:-1]
    return escaped

def process_doc(content, type='text'):
    """
    Convert a document to a JSON-compatible format.
    
    Args:
        input_path (str): Path to the input document.
        output_path (str, optional): Path to save the JSON output.
    
    Returns:
        dict: JSON-compatible dictionary with the document content.
    """

    # Preprocess based on file type
    if type == 'html':
        processed_content = preprocess_html(content)
    elif type == 'text':
        # Default to plain text processing (extendable for other formats)
        processed_content = preprocess_text(content)
    else:
        raise RuntimeError

    # Escape special characters for JSON
    escaped_content = escape_json_string(processed_content)

    return escaped_content


def main():
    # Example usage
    input_file = "/Users/liwenhao/Desktop/before.txt"
    output_file = "/Users/liwenhao/Desktop/after.txt"
    
    x = open(input_file, 'r').read()
    x = process_doc(x, type='text')
    open(output_file, 'w').write(x)

if __name__ == "__main__":
    main()