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
        'google': os.environ.get('GOOGLE_API_KEY'),
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

# OpenAI models
def get_valid_openai_models():
    """
    Get a list of valid OpenAI model names
    
    Returns:
        list: List of valid OpenAI model names
    """
    return [
        # Latest models
        'gpt-4-turbo',
        'gpt-4o',
        'gpt-4o-mini',
        'gpt-3.5-turbo',
        
        # Version-specific models
        'gpt-4-0613',
        'gpt-4-32k-0613',
        'gpt-4-1106-preview',
        'gpt-4-vision-preview',
        'gpt-3.5-turbo-0613',
        'gpt-3.5-turbo-16k-0613',
        'gpt-3.5-turbo-1106',
        
        # Legacy models
        'gpt-4',
        'gpt-4-32k',
        'gpt-3.5-turbo-16k'
    ]

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

def call_openai(model_name, prompt):
    """
    Call OpenAI's API with the given prompt using v0.28.0
    
    Args:
        model_name: Name of the OpenAI model to use
        prompt: Prompt to send to the model
        
    Returns:
        tuple: (response_text, prompt_tokens, completion_tokens, latency)
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
def test_models(models, prompt):
    """
    Test multiple models with the same prompt
    
    Args:
        models: List of model names to test
        prompt: Prompt to send to the models
        
    Returns:
        dict: Dictionary of model results
    """
    results = {}
    
    for model in models:
        print(f"\n📊 TESTING MODEL: {model}")
        
        start_time = time.time()
        try:
            if model.startswith('openai:') or not ':' in model:
                response, prompt_tokens, completion_tokens, latency = call_openai(model, prompt)
                success = not response.startswith("Error:")
            else:
                response = f"Error: Unsupported model provider: {model.split(':', 1)[0]}"
                prompt_tokens = 0
                completion_tokens = 0
                latency = 0
                success = False
                
            end_time = time.time()
            total_time = end_time - start_time
            
            results[model] = {
                'success': success,
                'response': response,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
                'latency': latency,
                'total_time': total_time
            }
            
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{status}: {model} ({total_time:.2f}s)")
            
        except Exception as e:
            end_time = time.time()
            total_time = end_time - start_time
            
            results[model] = {
                'success': False,
                'response': f"Error: {str(e)}",
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'latency': 0,
                'total_time': total_time
            }
            
            print(f"❌ FAILED: {model} ({total_time:.2f}s) - {str(e)}")
    
    return results
