import os
import time
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper functions
def load_api_keys():
    """
    Load API keys from environment variables or config file
    
    Returns:
        dict: Dictionary of API keys
    """
    keys = {
        'openai': os.environ.get('OPENAI_API_KEY'),
        'anthropic': os.environ.get('ANTHROPIC_API_KEY'),
        'google': os.environ.get('GEMINI_API_KEY'),  # Use GEMINI_API_KEY for Google/Gemini models
        'deepseek': os.environ.get('DEEPSEEK_API_KEY')
    }
    
    # Try to load from config file if environment variables are not set
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'api_keys' in config:
                    for provider, key in config['api_keys'].items():
                        if not keys.get(provider) and key:
                            keys[provider] = key
                            print(f"✅ Found API key for {provider}: {key[:5]}...") 
        except Exception as e:
            print(f"❌ Error loading config file: {str(e)}")
    
    # Check for placeholders and warn the user
    for provider, key in keys.items():
        if key and len(key) > 5:
            if provider == 'anthropic' and key.startswith('sk-ant-xxx'):
                print(f"⚠️ Warning: {provider} API key appears to be a placeholder: {key}")
                keys[provider] = None
            elif provider == 'google' and key.startswith('AIzaxxxxxx'):
                print(f"⚠️ Warning: {provider} API key appears to be a placeholder: {key}")
                keys[provider] = None
            elif provider == 'deepseek' and key.startswith('sk-xxxxxxx'):
                print(f"⚠️ Warning: {provider} API key appears to be a placeholder: {key}")
                keys[provider] = None
            elif provider == 'openai' and not (key.startswith('sk-') or key.startswith('org-')):
                print(f"⚠️ Warning: {provider} API key appears to be invalid: {key[:5]}...")
                keys[provider] = None
    
    # Report available keys
    for provider, key in keys.items():
        if key:
            print(f"✅ Found API key for {provider}: {key[:5]}...")
        else:
            print(f"❌ No valid API key found for {provider}")
    
    return keys

# Import centralized model configuration
from utils.model_config import (
    MODEL_DEFINITIONS, MODEL_PRICING,
    get_models_by_provider, get_model_pricing, calculate_cost
)

# OpenAI models
def get_valid_openai_models():
    """
    Get a list of valid OpenAI model names
    
    Returns:
        list: List of valid OpenAI model names
    """
    # Use the centralized configuration to get OpenAI models
    return get_models_by_provider('openai')

def validate_openai_model(model_name):
    """
    Validate the OpenAI model name and suggest an alternative if invalid
    
    Args:
        model_name: Name of the OpenAI model to validate
        
    Returns:
        tuple: (is_valid, suggested_model)
    """
    valid_models = get_valid_openai_models()
    
    if model_name in valid_models:
        return True, None
    
    # Handle common model name variants
    if model_name == "gpt4" or model_name == "gpt-4" or model_name.startswith("gpt-4-"):
        return False, "gpt-4-turbo"  # Suggest the latest GPT-4 model
    
    if model_name == "gpt-4o-latest" or model_name == "gpt4o":
        return False, "gpt-4o"  # Suggest the latest GPT-4o model
        
    if model_name == "gpt3.5" or model_name == "gpt-3.5" or model_name.startswith("gpt-3.5-"):
        return False, "gpt-3.5-turbo"  # Suggest the latest GPT-3.5 model
    
    # Default to GPT-4-turbo as fallback
    return False, "gpt-4-turbo"

def count_tokens(text, model_name):
    """
    Count tokens accurately using tiktoken or anthropic's tokenizer based on model
    
    Args:
        text: The text to count tokens for
        model_name: The model to use for tokenization
        
    Returns:
        int: Number of tokens in the text
    """
    # Check if this is a Claude model
    if model_name.startswith('claude-'):
        try:
            # Import anthropic library for Claude models
            import anthropic
            # Use anthropic's tokenizer for Claude models
            from anthropic import Anthropic
            client = Anthropic(api_key='placeholder')  # We'll set the real key in the API call
            # Count tokens using anthropic's tokenizer
            token_count = client.count_tokens(text)
            return token_count
        except ImportError:
            # If anthropic library is not available, use approximate method
            print("Warning: anthropic library not available, using approximate token count")
            # Claude models use a similar tokenizer to GPT models, rough estimate is sufficient
            return len(text.split()) * 1.3  # Rough approximation for Claude tokens
        except Exception as e:
            print(f"Error counting tokens for Claude model: {e}")
            return len(text.split()) * 1.3  # Fallback to rough approximation
    
    # For OpenAI models, use tiktoken
    import tiktoken
    
    # Get the appropriate encoding for the model
    try:
        if model_name.startswith('gpt-4'):
            encoding = tiktoken.encoding_for_model('gpt-4')
        elif model_name.startswith('gpt-3.5'):
            encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
        elif model_name.startswith('o1') or model_name.startswith('o3') or model_name.startswith('o4'):
            encoding = tiktoken.encoding_for_model('gpt-4')  # Use gpt-4 encoding for o1/o3/o4 models
        else:
            encoding = tiktoken.get_encoding('cl100k_base')  # Default encoding
        
        # Count tokens
        token_count = len(encoding.encode(text))
        return token_count
    except Exception as e:
        print(f"Error counting tokens: {e}")
        # Fallback method - rough estimate
        return len(text.split()) * 1.3  # Rough approximation

def call_anthropic_api(model_name, prompt, max_tokens=500, temperature=0.1):
    """
    Call the Anthropic API with error handling
    
    Args:
        model_name: Name of the Anthropic model to use
        prompt: The prompt to send to the API
        max_tokens: Maximum number of tokens to generate in the response
        temperature: Controls randomness (0-1)
        
    Returns:
        tuple: (response text, prompt tokens, completion tokens, latency)
    """
    import requests
    import json
    import time
    
    start_time = time.time()
    
    try:
        # Get API key from config.json
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
        api_key = None
        
        # Try to load from config file
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as config_file:
                    config = json.load(config_file)
                    api_key = config.get('api_keys', {}).get('anthropic', '')
        except Exception as config_error:
            print(f"Error loading config file: {config_error}")
        
        if not api_key:
            return "Error: Anthropic API key not found in config file.", 0, 0, 0
        
        # Set up API request
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        # Prepare payload for Messages API
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Make API call with retry logic
        max_retries = 3
        retry_delay = 1
        
        for retry in range(max_retries):
            try:
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=60  # 60 seconds timeout
                )
                
                response.raise_for_status()  # Raise exception for 4XX/5XX responses
                result = response.json()
                
                # Extract response content
                content_blocks = result.get("content", [])
                response_text = ""
                
                # Concatenate text content from all content blocks
                for block in content_blocks:
                    if block.get("type") == "text":
                        response_text += block.get("text", "")
                
                # Get usage information
                prompt_tokens = result.get("usage", {}).get("input_tokens", 0)
                completion_tokens = result.get("usage", {}).get("output_tokens", 0)
                
                # If usage is not provided, estimate using our token counter
                if prompt_tokens == 0:
                    prompt_tokens = count_tokens(prompt, model_name)
                if completion_tokens == 0:
                    completion_tokens = count_tokens(response_text, model_name)
                
                latency = time.time() - start_time
                return response_text, prompt_tokens, completion_tokens, latency
                
            except requests.exceptions.Timeout:
                print(f"Request timed out, retrying ({retry+1}/{max_retries})...")
                if retry < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                continue
                
            except requests.exceptions.HTTPError as e:
                print(f"HTTP error: {e}")
                if 500 <= e.response.status_code < 600 and retry < max_retries - 1:
                    print(f"Retrying on server error ({retry+1}/{max_retries})...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    error_detail = e.response.json() if e.response.content else {}
                    error_message = error_detail.get("error", {}).get("message", str(e))
                    return f"Error: API returned error: {error_message}", 0, 0, time.time() - start_time
                    
            except Exception as api_error:
                error_msg = str(api_error)
                print(f"❌ Error during API call: {error_msg}")
                return f"Error: API call failed: {error_msg}", 0, 0, time.time() - start_time
    
    except Exception as e:
        print(f"❌ ANTHROPIC API CALL FAILED: {model_name}")
        print(f"⏱️ Failed after: {(time.time() - start_time):.2f}s")
        print(f"🚨 Error: {str(e)}")
        return f"Error: {str(e)}", 0, 0, time.time() - start_time

def call_claude(model_name, prompt):
    """
    Call Claude API with the given prompt
    
    Args:
        model_name: Name of the Claude model to use
        prompt: Prompt to send to the API
        
    Returns:
        tuple: (response text, prompt tokens, completion tokens, latency)
    """
    max_tokens = 500
    temperature = 0.1
    
    print(f"🔄 CLAUDE API CALL STARTED: {model_name}")
    response_text, prompt_tokens, completion_tokens, latency = call_anthropic_api(model_name, prompt, max_tokens, temperature)
    
    if not response_text.startswith("Error:"):
        print(f"✅ CLAUDE API CALL COMPLETED: {model_name}")
        print(f"⏱️ Time: {latency:.2f} seconds")
        print(f"🔤 Tokens: {prompt_tokens} prompt + {completion_tokens} completion = {prompt_tokens + completion_tokens} total")
        
        # Truncate long responses in the log for clarity
        if len(response_text) > 300:
            response_preview = response_text[:300] + "..."
        else:
            response_preview = response_text
        print(f"📝 Response: {response_preview}")
    else:
        print(f"❌ CLAUDE API CALL FAILED: {model_name}")
        print(f"⏱️ Time: {latency:.2f} seconds")
        print(f"🚨 Error: {response_text}")
    
    return response_text, prompt_tokens, completion_tokens, latency

def call_gemini_api(model_name, prompt, max_tokens=500, temperature=0.1):
    """
    Call Gemini API with the given prompt
    
    Args:
        model_name: Name of the Gemini model to use
        prompt: Prompt to send to the API
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        
    Returns:
        tuple: (response text, prompt tokens, completion tokens, latency)
    """
    start_time = time.time()
    
    try:
        # Load API keys
        api_keys = load_api_keys()
        gemini_key = api_keys.get('google')
        
        if not gemini_key:
            # Try to load from config file
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
            try:
                if os.path.exists(config_path):
                    with open(config_path, 'r') as config_file:
                        config = json.load(config_file)
                        gemini_key = config.get('api_keys', {}).get('google', '')
            except Exception as config_error:
                print(f"Error loading config file: {config_error}")
            
        if not gemini_key:
            return "Error: Gemini API key not found in config file.", 0, 0, 0
        
        # Import the Google Generative AI library
        try:
            import google.generativeai as genai
        except ImportError:
            return "Error: google-generativeai package not installed. Please install it with 'pip install google-generativeai'.", 0, 0, 0
        
        # Configure the API
        genai.configure(api_key=gemini_key)
        
        # Set up the model
        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Make API call with retry logic
        max_retries = 3
        retry_delay = 1
        
        for retry in range(max_retries):
            try:
                # Create the model
                model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
                
                # Generate content
                response = model.generate_content(prompt)
                
                # Extract response text
                response_text = response.text
                
                # Get token counts (Gemini API doesn't provide token counts directly)
                # We'll use our token counter to estimate
                prompt_tokens = count_tokens(prompt, model_name)
                completion_tokens = count_tokens(response_text, model_name)
                
                latency = time.time() - start_time
                return response_text, prompt_tokens, completion_tokens, latency
                
            except Exception as api_error:
                error_msg = str(api_error)
                print(f"❌ Error during Gemini API call (attempt {retry+1}/{max_retries}): {error_msg}")
                
                if retry < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    return f"Error: Gemini API call failed after {max_retries} attempts: {error_msg}", 0, 0, time.time() - start_time
    
    except Exception as e:
        print(f"❌ GEMINI API CALL FAILED: {model_name}")
        print(f"⏱️ Failed after: {(time.time() - start_time):.2f}s")
        print(f"🚨 Error: {str(e)}")
        return f"Error: {str(e)}", 0, 0, time.time() - start_time

def call_gemini(model_name, prompt):
    """
    Call Gemini API with the given prompt
    
    Args:
        model_name: Name of the Gemini model to use
        prompt: Prompt to send to the API
        
    Returns:
        tuple: (response text, prompt tokens, completion tokens, latency)
    """
    max_tokens = 500
    temperature = 0.1
    
    print(f"🔄 GEMINI API CALL STARTED: {model_name}")
    response_text, prompt_tokens, completion_tokens, latency = call_gemini_api(model_name, prompt, max_tokens, temperature)
    
    if not response_text.startswith("Error:"):
        print(f"✅ GEMINI API CALL COMPLETED: {model_name}")
        print(f"⏱️ Time: {latency:.2f} seconds")
        print(f"🔤 Tokens: {prompt_tokens} prompt + {completion_tokens} completion = {prompt_tokens + completion_tokens} total")
        
        # Truncate long responses in the log for clarity
        if len(response_text) > 300:
            response_preview = response_text[:300] + "..."
        else:
            response_preview = response_text
        print(f"📝 Response: {response_preview}")
    else:
        print(f"❌ GEMINI API CALL FAILED: {model_name}")
        print(f"⏱️ Time: {latency:.2f} seconds")
        print(f"🚨 Error: {response_text}")
    
    return response_text, prompt_tokens, completion_tokens, latency

def call_openai(model_name, prompt):
    """
    Call OpenAI's API with the given prompt using v0.28.0
    
    Args:
        model_name: Name of the OpenAI model to use
        prompt: Prompt to send to the API
        
    Returns:
        tuple: (response text, prompt tokens, completion tokens, latency)
    """
    print(f"🔄 OPENAI API CALL STARTED: {model_name}")
    start_time = time.time()
    
    try:
        # Load API keys
        api_keys = load_api_keys()
        openai_key = api_keys.get('openai')
        
        if not openai_key:
            print(f"❌ ERROR: No OpenAI API key found in configuration")
            return "Error: No valid OpenAI API key found", 0, 0, 0
        
        print(f"🔍 DEBUG: OpenAI API Key (first 5 chars): {openai_key[:5]}...")
        
        # Set API key for OpenAI v0.28.0
        import openai
        openai.api_key = openai_key
        
        # Clean model name if needed
        model_short_name = model_name.replace('openai:', '')
        print(f"📤 Sending request to OpenAI API for model: {model_short_name}")
        
        # Validate the model name
        is_valid, suggested_model = validate_openai_model(model_short_name)
        if not is_valid and suggested_model:
            print(f"⚠️ WARNING: {model_short_name} is not a standard OpenAI model name.")
            print(f"🔄 Using suggested model: {suggested_model}")
            model_short_name = suggested_model
        
        # Construct the messages
        print(f"📝 Creating chat message with prompt: {prompt[:50]}...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes customer reviews."},
            {"role": "user", "content": prompt}
        ]
        
        # Make the API call using OpenAI v0.28.0
        print(f"🚀 Sending request to OpenAI API...")
        try:
            response = openai.ChatCompletion.create(
                model=model_short_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2048
            )
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Extract response data
            response_text = response['choices'][0]['message']['content']
            prompt_tokens = response['usage']['prompt_tokens']
            completion_tokens = response['usage']['completion_tokens']
            
            print(f"✅ OPENAI API CALL SUCCESS: {model_short_name}")
            print(f"⏱️ Latency: {latency:.2f}s")
            print(f"🔤 Tokens: {prompt_tokens} prompt, {completion_tokens} completion")
            print(f"📝 Response first 100 chars: {response_text[:100]}...")
            
            return response_text, prompt_tokens, completion_tokens, latency
            
        except Exception as api_error:
            error_msg = str(api_error)
            print(f"❌ Error during API call: {error_msg}")
            return f"Error: API call failed: {error_msg}", 0, 0, time.time() - start_time
    
    except Exception as e:
        print(f"❌ OPENAI API CALL FAILED: {model_name}")
        print(f"⏱️ Failed after: {(time.time() - start_time):.2f}s")
        print(f"🚨 Error: {str(e)}")
        return f"Error: {str(e)}", 0, 0, time.time() - start_time

# Function to test all models
def test_models(reviews, models, prompt_template):
    """
    Test multiple models with multiple reviews
    
    Args:
        reviews: List of review texts to process
        models: List of model names to test
        prompt_template: Template with {review} placeholder
        
    Returns:
        dict: Dictionary of model results and statistics
    """
    # Initialize results dictionary
    results = {}
    
    # Process each model
    for model in models:
        print(f"\n📊 TESTING MODEL: {model}")
        model_results = []
        successful_calls = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_latency = 0
        errors = 0
        
        # Process each review with this model
        for i, review in enumerate(reviews):
            # Replace the {review} placeholder in the template
            prompt = prompt_template.replace('{review}', review)
            # Store the complete prompt for reporting purposes
            full_prompt = prompt
            
            print(f"  - Processing review {i+1}/{len(reviews)}...")
            
            # Call the model
            start_time = time.time()
            try:
                # Check if this is a Claude model, Gemini model, or OpenAI model and call appropriate API
                if model.startswith('claude-'):
                    response, prompt_tokens, completion_tokens, latency = call_claude(model, prompt)
                elif model.startswith('gemini-'):
                    response, prompt_tokens, completion_tokens, latency = call_gemini(model, prompt)
                else:
                    # Default to OpenAI for all other models
                    response, prompt_tokens, completion_tokens, latency = call_openai(model, prompt)
                    
                end_time = time.time()
                total_time = end_time - start_time

                # Check if response was successful
                success = not response.startswith("Error:")

                if success:
                    successful_calls += 1
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    total_latency += latency
                else:
                    errors += 1
                
                # Calculate cost using the centralized pricing calculation function
                total_cost = calculate_cost(model.replace('openai:', ''), prompt_tokens, completion_tokens)
                
                # Store token counts for clear reporting in the results
                
                # Record result for this review
                result = {
                    'review': review[:100] + '...' if len(review) > 100 else review,
                    'full_review': review,  # Store the complete review
                    'full_prompt': full_prompt,  # Store the complete prompt that was sent
                    'success': success,
                    'response': response,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens,
                    'latency': latency,
                    'total_time': total_time,
                    'cost': total_cost
                }
                
                model_results.append(result)
                
                status = "✅" if success else "❌"
                print(f"    {status} Review {i+1}: {total_time:.2f}s, {prompt_tokens+completion_tokens} tokens")
                
            except Exception as e:
                end_time = time.time()
                total_time = end_time - start_time
                errors += 1
                
                # Record error result
                result = {
                    'review': review[:100] + '...' if len(review) > 100 else review,
                    'full_review': review,  # Store the complete review
                    'full_prompt': full_prompt,  # Store the complete prompt that was sent
                    'success': False,
                    'response': f"Error: {str(e)}",
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0,
                    'latency': 0,
                    'total_time': total_time,
                    'cost': 0.0
                }
                
                model_results.append(result)
                print(f"    ❌ Review {i+1}: {total_time:.2f}s - Error: {str(e)}")
        
        # Store all results for this model
        results[model] = model_results
        
        # Add summary statistics
        results[f"{model}_stats"] = {
            'successful': successful_calls,
            'errors': errors,
            'total_reviews': len(reviews),
            'total_prompt_tokens': total_prompt_tokens,
            'total_completion_tokens': total_completion_tokens,
            'total_tokens': total_prompt_tokens + total_completion_tokens,
            'avg_prompt_tokens': total_prompt_tokens / successful_calls if successful_calls > 0 else 0,
            'avg_completion_tokens': total_completion_tokens / successful_calls if successful_calls > 0 else 0,
            'avg_tokens': (total_prompt_tokens + total_completion_tokens) / successful_calls if successful_calls > 0 else 0,
            'average_latency': total_latency / successful_calls if successful_calls > 0 else 0
        }
        
        # Print summary for this model
        print(f"  Summary for {model}: {successful_calls}/{len(reviews)} successful, {errors} errors")
    
    return results
