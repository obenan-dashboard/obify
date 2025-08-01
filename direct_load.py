#!/usr/bin/env python3
import os
import json
import time
from datetime import datetime

# Define paths
UPLOAD_FOLDER = 'data/uploads'
REVIEWS_FILE = '/Users/jawadali/Desktop/reviews.csv'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Process the reviews file
def process_reviews_file(filepath):
    reviews = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            text = line.strip()
            if text and not text.lower().startswith('review'):
                reviews.append(text)
    return reviews

# Main function
def main():
    print(f"Processing reviews file: {REVIEWS_FILE}")
    
    # Extract reviews
    reviews = process_reviews_file(REVIEWS_FILE)
    print(f"Extracted {len(reviews)} reviews from file")
    
    # Create temp file with timestamp
    timestamp = int(time.time())
    temp_path = os.path.join(UPLOAD_FOLDER, f'temp_{timestamp}.json')
    
    # Save reviews to temp file
    with open(temp_path, 'w') as f:
        json.dump({
            'reviews': reviews, 
            'timestamp': timestamp, 
            'original_file': 'reviews.csv'
        }, f)
    
    print(f"Saved reviews to temporary file: {temp_path}")
    print(f"\nYou can now access the configuration page at:")
    print(f"http://127.0.0.1:5000/configure/{timestamp}")

if __name__ == "__main__":
    main()
