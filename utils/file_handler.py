import os
import csv
import json
import pandas as pd
import traceback
from werkzeug.utils import secure_filename
from typing import Dict, List, Tuple, Any

def save_uploaded_file(file, upload_folder, filename):
    """
    Save an uploaded file to the specified folder
    
    Args:
        file: FileStorage object from request.files
        upload_folder: Directory to save the file
        filename: Secured filename
        
    Returns:
        str: Path to the saved file
    """
    print(f"Saving file: {filename} to {upload_folder}")
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)
    print(f"File saved successfully to {filepath}")
    return filepath

def inspect_file(filepath) -> Dict[str, Any]:
    """
    Inspect an uploaded file and return column information and sample data
    
    Args:
        filepath: Path to the uploaded file
        
    Returns:
        dict: Dictionary with columns list and sample data
    """
    file_ext = filepath.split('.')[-1].lower()
    
    try:
        # For CSV files
        if file_ext == 'csv':
            try:
                df = pd.read_csv(filepath)
                columns = df.columns.tolist()
                # Get sample data (first 3 rows)
                sample_data = {}
                for col in columns:
                    sample_data[col] = df[col].head(3).astype(str).tolist()
                return {
                    'columns': columns,
                    'sample_data': sample_data,
                    'file_type': 'csv'
                }
            except Exception as e:
                print(f"Error inspecting CSV with header: {str(e)}")
                # Try without header
                try:
                    df = pd.read_csv(filepath, header=None)
                    columns = [str(i) for i in range(df.shape[1])]
                    sample_data = {}
                    for i, col in enumerate(df.columns):
                        sample_data[str(i)] = df[col].head(3).astype(str).tolist()
                    return {
                        'columns': columns,
                        'sample_data': sample_data,
                        'file_type': 'csv',
                        'no_header': True
                    }
                except Exception as e2:
                    print(f"Error inspecting CSV without header: {str(e2)}")
                    return {
                        'columns': [],
                        'sample_data': {},
                        'file_type': 'csv',
                        'error': str(e2)
                    }
        
        # For Excel files
        elif file_ext in ['xls', 'xlsx']:
            try:
                df = pd.read_excel(filepath)
                columns = df.columns.tolist()
                sample_data = {}
                for col in columns:
                    sample_data[col] = df[col].head(3).astype(str).tolist()
                return {
                    'columns': columns,
                    'sample_data': sample_data,
                    'file_type': 'excel'
                }
            except Exception as e:
                print(f"Error inspecting Excel with header: {str(e)}")
                # Try without header
                try:
                    df = pd.read_excel(filepath, header=None)
                    columns = [str(i) for i in range(df.shape[1])]
                    sample_data = {}
                    for i, col in enumerate(df.columns):
                        sample_data[str(i)] = df[col].head(3).astype(str).tolist()
                    return {
                        'columns': columns,
                        'sample_data': sample_data,
                        'file_type': 'excel',
                        'no_header': True
                    }
                except Exception as e2:
                    print(f"Error inspecting Excel without header: {str(e2)}")
                    return {
                        'columns': [],
                        'sample_data': {},
                        'file_type': 'excel',
                        'error': str(e2)
                    }
        
        # For text files
        elif file_ext == 'txt':
            # Text files are treated as single-column data
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()][:10]  # Get up to 10 lines
                return {
                    'columns': ['text'],
                    'sample_data': {'text': lines[:3]},
                    'file_type': 'txt'
                }
            except Exception as e:
                return {
                    'columns': [],
                    'sample_data': {},
                    'file_type': 'txt',
                    'error': str(e)
                }
        else:
            return {
                'columns': [],
                'sample_data': {},
                'file_type': 'unknown',
                'error': f"Unsupported file extension: {file_ext}"
            }
    except Exception as e:
        print(f"Fatal error inspecting file: {str(e)}")
        return {
            'columns': [],
            'sample_data': {},
            'file_type': 'unknown',
            'error': str(e)
        }


def extract_reviews_from_column(filepath, column_name) -> List[str]:
    """
    Extract reviews from a specific column in an uploaded file
    
    Args:
        filepath: Path to the uploaded file
        column_name: Name of the column containing reviews
        
    Returns:
        list: List of reviews extracted from the specified column
    """
    file_ext = filepath.split('.')[-1].lower()
    
    try:
        # For CSV files
        if file_ext == 'csv':
            try:
                df = pd.read_csv(filepath)
                if column_name in df.columns:
                    reviews = df[column_name].dropna().astype(str).tolist()
                    # Clean the reviews
                    reviews = [r.strip() for r in reviews if r.strip() and not r.strip().lower().startswith('review')]
                    return reviews
                else:
                    # Try with numeric column if column_name is a string representation of a number
                    try:
                        col_idx = int(column_name)
                        if col_idx < len(df.columns):
                            reviews = df.iloc[:, col_idx].dropna().astype(str).tolist()
                            reviews = [r.strip() for r in reviews if r.strip() and not r.strip().lower().startswith('review')]
                            return reviews
                    except (ValueError, TypeError):
                        pass
                    return []
            except Exception as e:
                print(f"Error extracting from CSV: {str(e)}")
                return []
        
        # For Excel files
        elif file_ext in ['xls', 'xlsx']:
            try:
                df = pd.read_excel(filepath)
                if column_name in df.columns:
                    reviews = df[column_name].dropna().astype(str).tolist()
                    # Clean the reviews
                    reviews = [r.strip() for r in reviews if r.strip() and not r.strip().lower().startswith('review')]
                    return reviews
                else:
                    # Try with numeric column
                    try:
                        col_idx = int(column_name)
                        if col_idx < len(df.columns):
                            reviews = df.iloc[:, col_idx].dropna().astype(str).tolist()
                            reviews = [r.strip() for r in reviews if r.strip() and not r.strip().lower().startswith('review')]
                            return reviews
                    except (ValueError, TypeError):
                        pass
                    return []
            except Exception as e:
                print(f"Error extracting from Excel: {str(e)}")
                return []
        
        # For text files (single column)
        elif file_ext == 'txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Try to determine the delimiter
            if '\n\n' in content:
                # Empty line separates reviews
                reviews = content.split('\n\n')
            else:
                # Each line is a review
                reviews = content.split('\n')
            
            # Clean up the reviews
            reviews = [r.strip() for r in reviews if r.strip()]
            return reviews
        
        else:
            return []
    
    except Exception as e:
        print(f"Fatal error extracting reviews: {str(e)}")
        return []


def process_uploaded_file(filepath):
    """
    Process an uploaded file (CSV, TXT, XLS, or XLSX) and extract reviews
    
    Args:
        filepath: Path to the uploaded file
        
    Returns:
        list: List of reviews extracted from the file
    """
    file_ext = filepath.split('.')[-1].lower()
    
    if file_ext == 'csv':
        return process_csv(filepath)
    elif file_ext == 'txt':
        return process_txt(filepath)
    elif file_ext in ['xls', 'xlsx']:
        return process_excel(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")

def process_csv(filepath):
    """
    Process a CSV file and extract reviews
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        list: List of reviews extracted from the CSV
    """
    try:
        # First attempt - try with pandas with header inference
        print(f"Attempting to process CSV file: {filepath}")
        reviews = []
        
        # Method 1: Using pandas with header
        try:
            df = pd.read_csv(filepath)
            print(f"CSV columns found: {df.columns.tolist()}")
            
            # Try to identify review column based on common names
            review_column_names = [
                'review_text',  # Prioritize review_text first (your format)
                'review', 'text', 'comment', 'feedback', 'content',
                'description', 'reviews', 'comments', 'message'
            ]
            
            # First check for exact matches to prioritize review_text column
            review_column = None
            for col in df.columns:
                col_lower = col.lower()
                if col_lower == 'review_text':  # Exact match for your format
                    review_column = col
                    print(f"Found exact match for review_text column: {col}")
                    break
            
            # If not found, look for partial matches
            if review_column is None:
                for col in df.columns:
                    col_lower = col.lower()
                    if col_lower in review_column_names or any(name in col_lower for name in review_column_names):
                        review_column = col
                        break
            
            # If no match found, use last column as it's often the review text in CSV files
            if review_column is None and len(df.columns) > 0:
                review_column = df.columns[-1]
                
            if review_column:
                print(f"Using column '{review_column}' for reviews")
                reviews = df[review_column].dropna().astype(str).tolist()
                # Clean the reviews but preserve full text without truncation
                reviews = [r.strip() for r in reviews if r.strip() and not r.strip().lower().startswith('review')]
                print(f"Found {len(reviews)} reviews from '{review_column}' column")
                if reviews:
                    # If first review has quotes, remove them for all reviews
                    if reviews[0].startswith('"') and reviews[0].endswith('"'):
                        reviews = [r.strip('"') for r in reviews]
                    return reviews
        except Exception as e:
            print(f"Error with pandas header processing: {str(e)}")
        
        # Method 2: Try without headers
        try:
            df = pd.read_csv(filepath, header=None)
            print(f"Read CSV without header, shape: {df.shape}")
            
            # If only one column, use that
            if df.shape[1] == 1:
                reviews = df[0].dropna().astype(str).tolist()
                reviews = [r.strip() for r in reviews if r.strip() and not r.strip().lower().startswith('review')]
                if reviews:
                    return reviews
            
            # Try the last column as it often contains the review text
            reviews = df.iloc[:, -1].dropna().astype(str).tolist()
            reviews = [r.strip() for r in reviews if r.strip() and not r.strip().lower().startswith('review')]
            if reviews:
                return reviews
        except Exception as e:
            print(f"Error with pandas no-header processing: {str(e)}")
        
        # Method 3: Handle alternating format (odd rows = labels, even rows = reviews)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                reviews = []
                for i, line in enumerate(lines):
                    text = line.strip()
                    if text and not text.lower().startswith('review'):
                        reviews.append(text)
                if reviews:
                    return reviews
        except Exception as e:
            print(f"Error with alternating format processing: {str(e)}")
        
        # Method 4: Last resort - try standard CSV module
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                all_rows = list(reader)
                
                # If we have more than one column, try the last one
                if all_rows and len(all_rows[0]) > 1:
                    reviews = [row[-1] for row in all_rows if row and row[-1].strip() and not row[-1].strip().lower().startswith('review')]
                    if reviews:
                        return reviews
                
                # Otherwise, use any non-empty value in each row
                reviews = []
                for row in all_rows:
                    for cell in row:
                        if cell and cell.strip() and not cell.strip().lower().startswith('review'):
                            reviews.append(cell.strip())
                            break
                if reviews:
                    return reviews
        except Exception as e:
            print(f"Error with CSV module processing: {str(e)}")
        
        # If we've tried everything and found nothing, return an empty list
        return []
    except Exception as e:
        print(f"Fatal error processing CSV: {str(e)}")
        return []

def process_txt(filepath):
    """
    Process a TXT file and extract reviews
    
    Args:
        filepath: Path to the TXT file
        
    Returns:
        list: List of reviews extracted from the TXT file
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Try to determine the delimiter (new line or other character)
    if '\n\n' in content:
        # Empty line separates reviews
        reviews = content.split('\n\n')
    else:
        # Each line is a review
        reviews = content.split('\n')
    
    # Clean up the reviews
    reviews = [r.strip() for r in reviews if r.strip()]
    return reviews

def process_excel(filepath):
    """
    Process an Excel file (XLS or XLSX) and extract reviews
    
    Args:
        filepath: Path to the Excel file
        
    Returns:
        list: List of reviews extracted from the Excel file
    """
    try:
        print(f"Processing Excel file: {filepath}")
        reviews = []
        
        # Read the Excel file
        try:
            df = pd.read_excel(filepath)
            print(f"Excel columns found: {df.columns.tolist()}")
            
            # Case 1: Check for format with review_text column
            review_column_names = [
                'review_text',  # Standard format
                'review', 'reviews', 'text', 'comment', 'comments', 'feedback', 'content',
                'description', 'message'
            ]
            
            # First check for exact matches
            review_column = None
            for col in df.columns:
                col_lower = col.lower()
                if col_lower == 'review_text':
                    review_column = col
                    print(f"Found standard CSV-like format with review_text column: {col}")
                    break
            
            # If not found, look for partial matches
            if review_column is None:
                for col in df.columns:
                    col_lower = col.lower()
                    if col_lower in review_column_names or any(name in col_lower for name in review_column_names):
                        review_column = col
                        print(f"Found review column: {col}")
                        break
            
            # Case 2: Check for format with just a 'Reviews' heading
            if review_column is None:
                # Check if we have a single column with 'Reviews' as header or first value
                if len(df.columns) == 1:
                    if df.columns[0].lower() == 'reviews':
                        print("Found single column with 'Reviews' header")
                        reviews = df[df.columns[0]].dropna().astype(str).tolist()
                        # Clean up the reviews
                        reviews = [r.strip() for r in reviews if r.strip() and not r.strip().lower() == 'reviews']
                        if reviews:
                            return reviews
                    else:
                        # Check if first row might be 'Reviews'
                        try:
                            if df.iloc[0, 0].lower() == 'reviews':
                                print("Found 'Reviews' as first row")
                                reviews = df[df.columns[0]].iloc[1:].dropna().astype(str).tolist()
                                # Clean up the reviews
                                reviews = [r.strip() for r in reviews if r.strip()]
                                if reviews:
                                    return reviews
                        except:
                            pass
            
            # If we found a review column earlier, use it
            if review_column:
                reviews = df[review_column].dropna().astype(str).tolist()
                # Clean up the reviews
                reviews = [r.strip() for r in reviews if r.strip() and not r.strip().lower() == 'reviews']
                print(f"Found {len(reviews)} reviews from '{review_column}' column")
                if reviews:
                    return reviews
            
            # If no specific review column found, try the last column as fallback
            if not reviews and len(df.columns) > 0:
                last_col = df.columns[-1]
                print(f"Trying last column as fallback: {last_col}")
                reviews = df[last_col].dropna().astype(str).tolist()
                reviews = [r.strip() for r in reviews if r.strip() and not r.strip().lower() == 'reviews']
                if reviews:
                    return reviews
        
        except Exception as e:
            print(f"Error with pandas Excel processing: {str(e)}")
            traceback.print_exc()
        
        # Try an alternative approach with no header
        try:
            df = pd.read_excel(filepath, header=None)
            print(f"Read Excel without header, shape: {df.shape}")
            
            # If only one column, use that, skipping any row that might be 'Reviews'
            if df.shape[1] == 1:
                reviews = df[0].dropna().astype(str).tolist()
                reviews = [r.strip() for r in reviews if r.strip() and not r.strip().lower() == 'reviews']
                if reviews:
                    return reviews
            
            # Try the last column as it often contains the review text
            reviews = df.iloc[:, -1].dropna().astype(str).tolist()
            reviews = [r.strip() for r in reviews if r.strip() and not r.strip().lower() == 'reviews']
            if reviews:
                return reviews
        except Exception as e:
            print(f"Error with Excel no-header processing: {str(e)}")
        
        # If all methods fail, return empty list
        return []
        
    except Exception as e:
        print(f"Fatal error processing Excel file: {str(e)}")
        traceback.print_exc()
        return []
