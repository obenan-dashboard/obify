import os
import json
import time
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, session
from werkzeug.utils import secure_filename
from datetime import datetime
import logging
import traceback

# Import utility modules
from utils.file_handler import process_uploaded_file, save_uploaded_file, inspect_file, extract_reviews_from_column
from utils.model_handler import call_openai, test_models  # Focus only on OpenAI models
from utils.openai_validator import validate_openai_key
from utils.evaluator import evaluate_results
from utils.report_generator import generate_report

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'data/uploads'
app.config['REPORTS_FOLDER'] = 'data/reports'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'txt', 'xls', 'xlsx'}

# Ensure directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['REPORTS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Validate API key at startup
try:
    logger.info("Validating OpenAI API key at startup...")
    
    # Load API keys from config
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            openai_key = config.get('api_keys', {}).get('openai')
    else:
        openai_key = os.environ.get('OPENAI_API_KEY')
        
    logger.info(f"Testing OpenAI API key: {openai_key[:5]}...")
    is_valid, error_message, available_models = validate_openai_key(openai_key)
    
    if not is_valid:
        logger.error(f"OpenAI API Key Validation Failed: {error_message}")
        logger.error("The application will run, but model testing may fail until a valid API key is provided.")
        app.config['API_KEY_VALID'] = False
        app.config['API_KEY_ERROR'] = error_message
        app.config['AVAILABLE_MODELS'] = []
    else:
        logger.info("OpenAI API Key Validation Successful")
        logger.info(f"Found {len(available_models)} available models")
        for i, model in enumerate(available_models[:5]):
            logger.info(f"  - {model}")
        if len(available_models) > 5:
            logger.info(f"  - ... and {len(available_models) - 5} more")
            
        app.config['API_KEY_VALID'] = True
        app.config['API_KEY_ERROR'] = None
        
        # Filter to only include the requested GPT models
        # Only include specific models requested by the user
        preferred_openai_models = [
            'gpt-4',
            'gpt-4-0613',
            'gpt-4o',
            'gpt-4.1',
            'gpt-4.1-mini',
            'gpt-4o-mini',
            'gpt-3.5-turbo'
        ]
        
        # Add Claude models
        claude_models = [
            'claude-3-7-sonnet-20250219',
            'claude-3-5-sonnet-20241022',
            'claude-3-5-haiku-20241022',
            'claude-3-5-sonnet-20240620',
            'claude-3-haiku-20240307',
            'claude-3-opus-20240229',
            'claude-3-sonnet-20240229',
            'claude-2.1',
            'claude-2.0'
        ]
        
        # Add Gemini models
        gemini_models = [
            'gemini-2.5-pro',
            'gemini-2.5-flash',
            'gemini-2.5-flash-lite',
            'gemini-2.0-flash',
            'gemini-2.0-flash-lite'
        ]
        
        # Format models for the template
        formatted_models = {
            'openai': [],
            'claude': [],
            'gemini': []
        }
        
        # First add preferred OpenAI models that are available
        for model_id in preferred_openai_models:
            if model_id in available_models:
                formatted_models['openai'].append(
                    {'id': model_id, 'name': model_id.replace('gpt-', 'GPT-').title()}
                )
                
        # Add Claude models
        for model_id in claude_models:
            # Anthropic models don't need to be validated like OpenAI models
            # Format the display name more nicely
            display_name = model_id
            if '-2025' in model_id or '-2024' in model_id:
                # Remove the date from the display name
                base_name = model_id.split('-2')[0]
                display_name = base_name.replace('claude-', 'Claude ').replace('-', ' ').title()
            else:
                display_name = model_id.replace('claude-', 'Claude ').title()
                
            formatted_models['claude'].append(
                {'id': model_id, 'name': display_name}
            )
            
        # Add Gemini models
        for model_id in gemini_models:
            # Format the display name more nicely
            display_name = model_id.replace('gemini-', 'Gemini ').replace('-', ' ').title()
            
            formatted_models['gemini'].append(
                {'id': model_id, 'name': display_name}
            )
        app.config['AVAILABLE_MODELS'] = formatted_models
except Exception as e:
    logger.error(f"Error during API key validation: {str(e)}")
    app.config['API_KEY_VALID'] = False
    app.config['API_KEY_ERROR'] = str(e)
    app.config['AVAILABLE_MODELS'] = {
        'openai': [
            {'id': 'gpt-3.5-turbo', 'name': 'GPT-3.5 Turbo'},
            {'id': 'gpt-4-turbo', 'name': 'GPT-4 Turbo'},
            {'id': 'gpt-4o', 'name': 'GPT-4o'},
            {'id': 'gpt-4o-mini', 'name': 'GPT-4o Mini'}
        ],
        'claude': [
            {'id': 'claude-3-7-sonnet-20250219', 'name': 'Claude 3.7 Sonnet'},
            {'id': 'claude-3-5-sonnet-20241022', 'name': 'Claude 3.5 Sonnet'},
            {'id': 'claude-3-5-haiku-20241022', 'name': 'Claude 3.5 Haiku'},
            {'id': 'claude-3-opus-20240229', 'name': 'Claude 3 Opus'},
            {'id': 'claude-3-haiku-20240307', 'name': 'Claude 3 Haiku'}
        ],
        'gemini': [
            {'id': 'gemini-2.5-pro', 'name': 'Gemini 2.5 Pro'},
            {'id': 'gemini-2.5-flash', 'name': 'Gemini 2.5 Flash'},
            {'id': 'gemini-2.5-flash-lite', 'name': 'Gemini 2.5 Flash Lite'},
            {'id': 'gemini-2.0-flash', 'name': 'Gemini 2.0 Flash'},
            {'id': 'gemini-2.0-flash-lite', 'name': 'Gemini 2.0 Flash Lite'}
        ]
    }

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_available_models():
    """
    Get available OpenAI models structured for the UI
    
    Returns:
        dict: Dictionary of models grouped by provider
    """
    # First check if we already validated and stored models in the app config
    if app.config.get('AVAILABLE_MODELS'):
        logger.info(f"Using cached models from app config")
        return app.config['AVAILABLE_MODELS']
    
    # If we don't have models stored, use a predefined list of supported models
    logger.info("No cached models found, using predefined OpenAI models list")
    
    # Structure the models as expected by the template
    return {
        'openai': [
            {'id': 'gpt-3.5-turbo', 'name': 'GPT-3.5 Turbo'},
            {'id': 'gpt-4-turbo', 'name': 'GPT-4 Turbo'},
            {'id': 'gpt-4o', 'name': 'GPT-4o'},
            {'id': 'gpt-4o-mini', 'name': 'GPT-4o Mini'}
        ],
        'claude': [
            {'id': 'claude-3-7-sonnet-20250219', 'name': 'Claude 3.7 Sonnet'},
            {'id': 'claude-3-5-sonnet-20241022', 'name': 'Claude 3.5 Sonnet'},
            {'id': 'claude-3-5-haiku-20241022', 'name': 'Claude 3.5 Haiku'},
            {'id': 'claude-3-opus-20240229', 'name': 'Claude 3 Opus'},
            {'id': 'claude-3-haiku-20240307', 'name': 'Claude 3 Haiku'}
        ]
    }

# Routes
@app.route('/')
def index():
    models = get_available_models()
    logger.info("Index route called")
    return render_template('index.html', models=models)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    logger.info(f"Upload route called with method: {request.method}")
    logger.info(f"Request form: {request.form}")
    logger.info(f"Request files: {request.files}")
    
    if request.method == 'GET':
        logger.info("Redirecting GET request to index")
        return redirect(url_for('index'))
        
    if 'file' not in request.files:
        logger.error("No file part in request")
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    logger.info(f"Received file: {file.filename}")
    
    if file.filename == '':
        logger.error("No file selected")
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        logger.info(f"Processing valid file: {file.filename}")
        filename = secure_filename(file.filename)
        try:
            filepath = save_uploaded_file(file, app.config['UPLOAD_FOLDER'], filename)
            logger.info(f"File saved at: {filepath}")
            
            # Inspect the file to get column information
            file_info = inspect_file(filepath)
            logger.info(f"File inspection complete. Found {len(file_info.get('columns', []))} columns")
            
            # Store file info in temporary file for next step
            timestamp = int(time.time())
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{timestamp}.json')
            
            # Add filepath to file_info for later use
            file_info['filepath'] = filepath
            file_info['original_filename'] = filename
            file_info['timestamp'] = timestamp
            
            with open(temp_path, 'w') as f:
                json.dump(file_info, f)
            
            # If there's only one column, extract reviews directly and proceed
            if len(file_info.get('columns', [])) == 1:
                column_name = file_info['columns'][0]
                reviews = extract_reviews_from_column(filepath, column_name)
                logger.info(f"Auto-selected single column: {column_name}. Found {len(reviews)} reviews")
                
                # Store reviews for the configure page
                review_data = {
                    'reviews': reviews,
                    'timestamp': timestamp,
                    'original_file': filename,
                    'selected_column': column_name
                }
                
                review_path = os.path.join(app.config['UPLOAD_FOLDER'], f'reviews_{timestamp}.json')
                with open(review_path, 'w') as f:
                    json.dump(review_data, f)
                
                logger.info(f"Single column auto-selected. Redirecting to configure_test with timestamp {timestamp}")
                return redirect(url_for('configure_test', timestamp=timestamp))
            else:
                # Multiple columns - redirect to column selection page
                logger.info(f"Multiple columns found. Redirecting to select_column with timestamp {timestamp}")
                return redirect(url_for('select_column', timestamp=timestamp))
                
        except Exception as e:
            logger.error(f"Error processing upload: {str(e)}")
            logger.error(traceback.format_exc())
            flash(f"Error processing file: {str(e)}")
            return redirect(url_for('index'))
    
    logger.error(f"Invalid file type: {file.filename}")
    flash('Invalid file type. Please upload a CSV, TXT, or Excel file (.xls, .xlsx)')
    return redirect(url_for('index'))


@app.route('/select_column/<timestamp>', methods=['GET', 'POST'])
def select_column(timestamp):
    logger.info(f"Select column route called with timestamp: {timestamp}")
    
    # Path to the temporary file
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{timestamp}.json')
    
    if not os.path.exists(temp_path):
        logger.error(f"Temporary file not found: {temp_path}")
        flash('Session expired or invalid')
        return redirect(url_for('index'))
    
    try:
        # Load the file info
        with open(temp_path, 'r') as f:
            file_info = json.load(f)
        
        if request.method == 'POST':
            # Process column selection
            selected_column = request.form.get('column')
            logger.info(f"Selected column: {selected_column}")
            
            if not selected_column:
                flash('Please select a column')
                return redirect(url_for('select_column', timestamp=timestamp))
            
            # Extract reviews from the selected column
            filepath = file_info.get('filepath')
            reviews = extract_reviews_from_column(filepath, selected_column)
            logger.info(f"Extracted {len(reviews)} reviews from column: {selected_column}")
            
            # Store the reviews for the configure page
            review_data = {
                'reviews': reviews,
                'timestamp': timestamp,
                'original_file': file_info.get('original_filename'),
                'selected_column': selected_column
            }
            
            review_path = os.path.join(app.config['UPLOAD_FOLDER'], f'reviews_{timestamp}.json')
            with open(review_path, 'w') as f:
                json.dump(review_data, f)
            
            logger.info(f"Redirecting to configure_test with timestamp {timestamp}")
            return redirect(url_for('configure_test', timestamp=timestamp))
        
        # Render column selection template
        logger.info(f"Rendering select_column template with {len(file_info.get('columns', []))} columns")
        return render_template(
            'select_column.html',
            columns=file_info.get('columns', []),
            sample_data=file_info.get('sample_data', {}),
            timestamp=timestamp,
            file_type=file_info.get('file_type', 'unknown'),
            original_filename=file_info.get('original_filename', 'Unknown file')
        )
        
    except Exception as e:
        logger.error(f"Error in select_column: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f"Error processing column selection: {str(e)}")
        return redirect(url_for('index'))

@app.route('/configure/<timestamp>')
def configure_test(timestamp):
    logger.info(f"Configure test route called with timestamp: {timestamp}")
    
    # Look for reviews data file first (from the new workflow)
    reviews_path = os.path.join(app.config['UPLOAD_FOLDER'], f'reviews_{timestamp}.json')
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{timestamp}.json')
    
    if os.path.exists(reviews_path):
        # New workflow - load from reviews file
        logger.info(f"Loading reviews from reviews_{timestamp}.json")
        with open(reviews_path, 'r') as f:
            data = json.load(f)
        
        original_file = data.get('original_file', 'Unknown file')
        selected_column = data.get('selected_column', 'Unknown column')
        logger.info(f"Loaded {len(data['reviews'])} reviews from file {original_file}, column {selected_column}")
    
    elif os.path.exists(temp_path):
        # Legacy fallback to old workflow
        logger.info(f"Reviews file not found, falling back to temp_{timestamp}.json (legacy mode)")
        with open(temp_path, 'r') as f:
            data = json.load(f)
        
        # Check if this is a new format temp file
        if 'reviews' not in data and 'filepath' in data:
            # This is a new format temp file but reviews file doesn't exist - redirect to selection
            logger.info("New format temp file found but reviews not extracted yet")
            flash("Please select a column containing reviews")
            return redirect(url_for('select_column', timestamp=timestamp))
    else:
        logger.error(f"Neither reviews_{timestamp}.json nor temp_{timestamp}.json found")
        flash('Session expired or invalid')
        return redirect(url_for('index'))
    
    # Get available models
    models = get_available_models()
    
    # Ensure we have reviews
    if 'reviews' not in data or not data['reviews']:
        logger.error("No reviews found in the data")
        flash('No reviews found in the uploaded file. Please try a different file or column.')
        return redirect(url_for('index'))
    
    all_reviews = data['reviews']  # Send all reviews to the template
    total_reviews = len(data['reviews'])
    
    original_filename = data.get('original_file', data.get('original_filename', 'Unknown file'))
    selected_column = data.get('selected_column', 'Auto-detected')
    
    logger.info(f"Rendering configure template with {len(models)} models and {total_reviews} reviews")
    return render_template(
        'configure.html', 
        models=models, 
        all_reviews=all_reviews,
        total_reviews=total_reviews,
        timestamp=timestamp,
        original_filename=original_filename,
        selected_column=selected_column
    )

@app.route('/run_test', methods=['POST'])
def run_test():
    logger.info(f"\n=====================================================================")
    logger.info(f"🚀 RUN TEST ROUTE CALLED with method: {request.method}")
    logger.info(f"=====================================================================")
    
    # Check API key validity first
    if not app.config.get('API_KEY_VALID', False):
        error_message = app.config.get('API_KEY_ERROR', 'Unknown error with API key')
        logger.error(f"❌ API KEY VALIDATION FAILED: {error_message}")
        flash(f'API Key Error: {error_message}. Please update your API key in config.json and restart the application.')
        return redirect(url_for('index'))
    
    # Re-validate API key in case it has been revoked or rate-limited
    try:
        # Load API keys from config
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                openai_key = config.get('api_keys', {}).get('openai')
        else:
            openai_key = os.environ.get('OPENAI_API_KEY')
            
        logger.info(f"Re-validating OpenAI API key: {openai_key[:5]}...")
        is_valid, error_message, _ = validate_openai_key(openai_key)
        
        if not is_valid:
            logger.error(f"❌ API KEY RE-VALIDATION FAILED: {error_message}")
            flash(f'API Key Error: {error_message}. Please update your API key in config.json and restart the application.')
            return redirect(url_for('index'))
        else:
            logger.info("✅ API key re-validation successful")
    except Exception as e:
        logger.error(f"❌ ERROR DURING API KEY RE-VALIDATION: {str(e)}")
        flash(f'API Key Validation Error: {str(e)}')
        return redirect(url_for('index'))
    
    # Process form data
    timestamp = request.form.get('timestamp')
    selected_models = request.form.getlist('models')
    prompt_template = request.form.get('promptTemplate')  # Changed from 'prompt_template' to 'promptTemplate' to match the HTML form field
    
    logger.info(f"🔑 FORM DATA DETAILS:")
    logger.info(f"   - Timestamp: {timestamp}")
    logger.info(f"   - Selected models: {selected_models}")
    logger.info(f"   - Prompt template length: {len(prompt_template) if prompt_template else 0}")
    logger.info(f"   - Prompt template preview: {prompt_template[:100]}..." if prompt_template else "No prompt template!")
    
    if not timestamp or not selected_models or not prompt_template:
        logger.error("❌ MISSING REQUIRED PARAMETERS:")
        logger.error(f"   - timestamp present: {timestamp is not None}")
        logger.error(f"   - selected_models present: {len(selected_models) > 0}")
        logger.error(f"   - prompt_template present: {prompt_template is not None}")
        flash('Missing required parameters')
        return redirect(url_for('index'))
    
    # Load the reviews
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{timestamp}.json')
    # New workflow: Check for reviews file first
    reviews_path = os.path.join(app.config['UPLOAD_FOLDER'], f'reviews_{timestamp}.json')
    logger.info(f"📂 Looking for reviews file at: {reviews_path}")
    
    if os.path.exists(reviews_path):
        # Use the new reviews file format
        logger.info(f"✅ Reviews file found, loading data from new format")
        try:
            with open(reviews_path, 'r') as f:
                data = json.load(f)
            
            reviews = data['reviews']
            original_filename = data.get('original_file', 'Unknown file')
            selected_column = data.get('selected_column', 'Unknown column')
            
            logger.info(f"📋 DATA LOADED FROM REVIEWS FILE:")
            logger.info(f"   - Original filename: {original_filename}")
            logger.info(f"   - Selected column: {selected_column}")
            logger.info(f"   - Number of reviews: {len(reviews)}")
            logger.info(f"   - First review: {reviews[0][:100]}..." if reviews else "No reviews!")
        except Exception as e:
            logger.error(f"❌ ERROR LOADING REVIEWS DATA: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            flash(f'Error loading review data: {str(e)}')
            return redirect(url_for('index'))
    else:
        # Fall back to old format
        logger.info(f"Reviews file not found, checking legacy temp file at: {temp_path}")
        
        if not os.path.exists(temp_path):
            logger.error(f"❌ TEMPORARY FILE NOT FOUND: {temp_path}")
            flash('Session expired or invalid')
            return redirect(url_for('index'))
        
        logger.info(f"✅ Legacy temp file found, loading data")
        try:
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            # Check if this is a new-format temp file that hasn't been processed
            if 'reviews' not in data and 'filepath' in data:
                logger.error(f"❌ New format temp file found but column not selected yet")
                flash('Please select a column containing reviews')
                return redirect(url_for('select_column', timestamp=timestamp))
            
            reviews = data['reviews']
            original_filename = data.get('original_file', 'Unknown file')
            
            logger.info(f"📋 DATA LOADED FROM LEGACY FORMAT:")
            logger.info(f"   - Original filename: {original_filename}")
            logger.info(f"   - Number of reviews: {len(reviews)}")
            logger.info(f"   - First review: {reviews[0][:100]}..." if reviews else "No reviews!")
        except Exception as e:
            logger.error(f"❌ ERROR LOADING LEGACY DATA: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            flash(f'Error loading review data: {str(e)}')
            return redirect(url_for('index'))
    # If we get here, one of the data loading paths succeeded
    
    # If there are too many reviews, limit to a reasonable number for testing
    if len(reviews) > 100:
        logger.info(f"⚙️ Limiting to 100 reviews for testing (out of {len(reviews)})")
        reviews = reviews[:100]
    else:
        logger.info(f"⚙️ Processing all {len(reviews)} reviews")

    
    # Run tests and get results
    try:
        logger.info(f"\n=====================================================================")
        logger.info(f"🧪 STARTING MODEL TESTING")
        logger.info(f"=====================================================================")
        logger.info(f"   - Reviews to process: {len(reviews)}")
        logger.info(f"   - Models to test: {', '.join(selected_models)}")
        
        for i, model in enumerate(selected_models):
            logger.info(f"🔍 MODEL {i+1}/{len(selected_models)}: {model}")
        
        results = test_models(reviews, selected_models, prompt_template)
        
        logger.info(f"\n=====================================================================")
        logger.info(f"✅ TEST MODELS COMPLETED")
        logger.info(f"=====================================================================")
        logger.info(f"   - Result keys: {list(results.keys())}")
        
        # Log detailed results for debugging
        for model_name in selected_models:
            model_results = results.get(model_name, [])
            stats_key = f"{model_name}_stats" 
            stats = results.get(stats_key, {})
            
            logger.info(f"📊 RESULTS FOR {model_name}:")
            logger.info(f"   - Number of results: {len(model_results)}")
            logger.info(f"   - Successes: {stats.get('successful', 0)} / {stats.get('total_reviews', 0)}")
            logger.info(f"   - Errors: {stats.get('errors', 0)}")
            logger.info(f"   - Avg tokens: {stats.get('avg_tokens', 0)}")
            logger.info(f"   - Avg latency: {stats.get('avg_latency', 0):.2f}s")
            
            # Log the first result detail for each model
            if model_results:
                first_result = model_results[0]
                if 'error' in first_result:
                    logger.info(f"   - First result has ERROR: {first_result['error'][:200]}...")
                else:
                    logger.info(f"   - First result tokens: {first_result.get('total_tokens', 0)}")
                    logger.info(f"   - First result latency: {first_result.get('latency', 0):.2f}s")
                    logger.info(f"   - First result response preview: {first_result.get('response', '')[:100]}...")
            else:
                logger.error(f"   - NO RESULTS FOUND FOR {model_name}")
    except Exception as e:
        logger.error(f"\n=====================================================================")
        logger.error(f"❌ ERROR DURING MODEL TESTING: {str(e)}")
        logger.error(f"=====================================================================")
        logger.error(f"Traceback: {traceback.format_exc()}")
        flash(f'Error testing models: {str(e)}')
        return redirect(url_for('index'))
    
    # Evaluate results
    try:
        logger.info(f"\n=====================================================================")
        logger.info(f"📈 STARTING RESULT EVALUATION")
        logger.info(f"=====================================================================")
        evaluation = evaluate_results(results)
        logger.info(f"✅ Evaluation completed with {len(evaluation)} evaluations")
        
        # Log evaluation results
        for model_name, eval_data in evaluation.items():
            logger.info(f"📊 EVALUATION FOR {model_name}:")
            for metric, value in eval_data.items():
                if isinstance(value, (int, float)):
                    logger.info(f"   - {metric}: {value}")
                else:
                    logger.info(f"   - {metric}: {str(value)[:50]}...")
    except Exception as e:
        logger.error(f"\n=====================================================================")
        logger.error(f"❌ ERROR DURING EVALUATION: {str(e)}")
        logger.error(f"=====================================================================")
        logger.error(f"Traceback: {traceback.format_exc()}")
        flash(f'Error evaluating results: {str(e)}')
        return redirect(url_for('index'))
    
    # Generate report
    try:
        logger.info(f"\n=====================================================================")
        logger.info(f"📝 GENERATING REPORT")
        logger.info(f"=====================================================================")
        report_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"   - Report name: {report_name}")
        report_path = generate_report(results, evaluation, original_filename, report_name, app.config['REPORTS_FOLDER'])
        logger.info(f"✅ Report generated at: {report_path}")
    except Exception as e:
        logger.error(f"\n=====================================================================")
        logger.error(f"❌ ERROR GENERATING REPORT: {str(e)}")
        logger.error(f"=====================================================================")
        logger.error(f"Traceback: {traceback.format_exc()}")
        flash(f'Error generating report: {str(e)}')
        return redirect(url_for('index'))
    
    # Clean up temp file
    try:
        os.remove(temp_path)
        logger.info(f"🧹 Temporary file removed: {temp_path}")
    except Exception as e:
        logger.error(f"⚠️ Error removing temp file: {str(e)}")
    
    logger.info(f"\n=====================================================================")
    logger.info(f"✅ TEST RUN COMPLETED SUCCESSFULLY")
    logger.info(f"   - Redirecting to report: {report_name}")
    logger.info(f"=====================================================================")
    
    return redirect(url_for('view_report', report_name=report_name))

@app.route('/report/<report_name>')
def view_report(report_name):
    logger.info(f"View report route called with report name: {report_name}")
    report_path = os.path.join(app.config['REPORTS_FOLDER'], f"{report_name}.json")
    
    if not os.path.exists(report_path):
        logger.error("Report not found")
        flash('Report not found')
        return redirect(url_for('index'))
    
    with open(report_path, 'r') as f:
        report_data = json.load(f)
    
    logger.info(f"Rendering report template with report data: {report_data}")
    return render_template('report.html', report=report_data, report_name=report_name)

@app.route('/api/download_report/<report_name>')
def download_report(report_name):
    logger.info(f"Download report route called with report name: {report_name}")
    report_path = os.path.join(app.config['REPORTS_FOLDER'], f"{report_name}.json")
    
    if not os.path.exists(report_path):
        logger.error("Report not found")
        return jsonify({'error': 'Report not found'}), 404
    
    return send_file(report_path, as_attachment=True)

@app.route('/api/export_report_pdf/<report_name>')
def export_report_pdf(report_name):
    logger.info(f"Export report PDF route called with report name: {report_name}")
    report_path = os.path.join(app.config['REPORTS_FOLDER'], f"{report_name}.json")
    pdf_path = os.path.join(app.config['REPORTS_FOLDER'], f"{report_name}.pdf")
    
    if not os.path.exists(report_path):
        logger.error("Report not found")
        return jsonify({'error': 'Report not found'}), 404
    
    try:
        # Load the report data
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        # Import ReportLab here to avoid loading it on startup
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
        from reportlab.lib.units import inch
        
        logger.info(f"Generating PDF for {report_name} using ReportLab")
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Custom styles
        title_style = ParagraphStyle(
            name='TitleStyle',
            parent=styles['Heading1'],
            fontSize=24,
            alignment=1,  # Center
            spaceAfter=12
        )
        subtitle_style = ParagraphStyle(
            name='SubtitleStyle',
            parent=styles['Heading2'],
            fontSize=14,
            alignment=1,  # Center
            textColor=colors.gray,
            spaceAfter=20
        )
        section_style = ParagraphStyle(
            name='SectionStyle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.blue,
            spaceAfter=12
        )
        normal_style = styles['Normal']
        bold_style = ParagraphStyle(
            name='BoldStyle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold'
        )
        
        # Title
        elements.append(Paragraph("Obify Analysis Report", title_style))
        timestamp = report_data.get('timestamp', '').replace('T', ' at ')[:19]
        elements.append(Paragraph(f"Generated on {timestamp}", subtitle_style))
        elements.append(Spacer(1, 0.5*inch))
        
        # Report overview
        elements.append(Paragraph("Report Overview", section_style))
        overview_data = [
            ["Source File", report_data.get('source_file', 'N/A')],
            ["Models Tested", str(len(report_data.get('models', {})))],
            ["Total Reviews", str(report_data.get('metrics', {}).get('total_reviews', 'N/A'))]
        ]
        
        if 'recommendations' in report_data:
            overview_data.extend([
                ["Best Overall", report_data['recommendations'].get('best_overall', 'N/A')],
                ["Best Value", report_data['recommendations'].get('best_value', 'N/A')],
                ["Fastest", report_data['recommendations'].get('fastest', 'N/A')],
                ["Most Accurate", report_data['recommendations'].get('most_accurate', 'N/A')]
            ])
            
        overview_table = Table(overview_data, colWidths=[2*inch, 3.5*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(overview_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Model Performance Summary
        elements.append(Paragraph("Model Performance Summary", section_style))
        
        # Header row for the summary table
        performance_data = [["Model", "Success Rate", "Avg. Latency", "Cost Per 1000", "Total Tokens"]]
        
        # Add data rows
        for model_id, model_data in report_data.get('models', {}).items():
            model_name = model_data.get('model_name', model_id)
            performance_data.append([
                f"{model_name}\n({model_id})",
                f"{model_data.get('success_rate', 'N/A')}%",
                f"{model_data.get('avg_latency', 'N/A')}s",
                f"${model_data.get('cost_per_1000', 'N/A')}",
                f"{model_data.get('total_tokens', 'N/A')}"
            ])
        
        # Create and style the table
        performance_table = Table(performance_data, colWidths=[2.2*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        performance_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(performance_table)
        elements.append(Spacer(1, 0.3*inch))

        # Detailed Model Analysis
        elements.append(Paragraph("Detailed Model Analysis", section_style))
        
        for model_id, model_data in report_data.get('models', {}).items():
            elements.append(Paragraph(f"{model_data.get('model_name', 'Unknown Model')}", bold_style))
            elements.append(Paragraph(f"Model ID: {model_id}", styles['Italic']))
            
            # Model metrics
            metrics_data = [
                ["Success Rate", "Avg. Latency", "Cost Per 1000"],
                [
                    f"{model_data.get('success_rate', 'N/A')}%",
                    f"{model_data.get('avg_latency', 'N/A')}s",
                    f"${model_data.get('cost_per_1000', 'N/A')}"
                ]
            ]
            
            metrics_table = Table(metrics_data, colWidths=[2*inch, 2*inch, 2*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ]))
            elements.append(metrics_table)
            
            # Token info
            elements.append(Spacer(1, 0.1*inch))
            token_data = [
                ["Input Tokens", f"{model_data.get('input_tokens', 'N/A')}", 
                 "Output Tokens", f"{model_data.get('output_tokens', 'N/A')}"],
                ["Avg Input", f"{model_data.get('avg_input_tokens', 'N/A')}", 
                 "Avg Output", f"{model_data.get('avg_output_tokens', 'N/A')}"],
            ]
            
            token_table = Table(token_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            token_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            elements.append(token_table)
            elements.append(Spacer(1, 0.1*inch))
            elements.append(Paragraph("Cost Formula: (Input ÷ 1M × Rate) + (Output ÷ 1M × Rate)", styles['Italic']))
            elements.append(Spacer(1, 0.3*inch))
        
        # Sample Reviews Section
        elements.append(PageBreak())
        elements.append(Paragraph("Sample Reviews & Responses", section_style))
        
        # Show up to 3 reviews for each model
        for model_id, model_data in report_data.get('models', {}).items():
            if model_id in report_data.get('all_responses', {}) and report_data['all_responses'][model_id]:
                elements.append(Paragraph(f"{model_data.get('model_name', model_id)}", bold_style))
                elements.append(Paragraph(f"Model ID: {model_id}", styles['Italic']))
                elements.append(Spacer(1, 0.1*inch))
                
                responses = report_data['all_responses'][model_id][:3]  # Show up to 3 responses
                
                for i, response in enumerate(responses):
                    elements.append(Paragraph(f"Review #{response.get('review_number', i+1)}", bold_style))
                    elements.append(Paragraph(response.get('review_text', 'N/A'), normal_style))
                    elements.append(Spacer(1, 0.1*inch))
                    elements.append(Paragraph("Response:", normal_style))
                    
                    # Split the response text into paragraphs to avoid overflow
                    resp_text = response.get('response', 'N/A')
                    for paragraph in resp_text.split('\n'):
                        if paragraph.strip():
                            elements.append(Paragraph(paragraph, styles['Code']))
                    
                    elements.append(Spacer(1, 0.2*inch))
                
                # Note if there are more responses
                if len(report_data['all_responses'][model_id]) > 3:
                    more_count = len(report_data['all_responses'][model_id]) - 3
                    elements.append(Paragraph(
                        f"... and {more_count} more responses available in the interactive report.",
                        styles['Italic']
                    ))
                
                elements.append(Spacer(1, 0.3*inch))
        
        # Footer
        elements.append(Spacer(1, inch))
        elements.append(Paragraph("Generated by Obify - AI Model Comparison Platform", 
                             ParagraphStyle(name='Footer', parent=normal_style, alignment=1)))
        elements.append(Paragraph(" 2025 Obify. All rights reserved.", 
                             ParagraphStyle(name='Footer', parent=normal_style, alignment=1)))
        
        # Build PDF
        doc.build(elements)
        
        # Check if PDF was created
        if not os.path.exists(pdf_path):
            logger.error(f"Failed to generate PDF at {pdf_path}")
            return jsonify({'error': 'Failed to generate PDF'}), 500
            
        logger.info(f"PDF successfully generated at {pdf_path}")
        return send_file(pdf_path, as_attachment=True, download_name=f"{report_name}.pdf")
        
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Failed to generate PDF: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5005)
    
    
   