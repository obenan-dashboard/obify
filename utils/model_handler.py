import os
import time
import json
import logging
from pathlib import Path
try:
    # Optional: load .env if present (no hard dependency)
    from dotenv import load_dotenv  # type: ignore
    _DOTENV_LOADED = False
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore
    _DOTENV_LOADED = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper functions
def load_api_keys():
    """Load API keys strictly from environment variables (.env supported).

    Priority/order:
      1. If python-dotenv is available and not yet loaded, attempt to load project root .env
      2. Read required keys from os.environ

    Returns:
        dict: provider -> key (missing or placeholder keys returned as None)
    """
    global _DOTENV_LOADED
    if load_dotenv and not _DOTENV_LOADED:
        # Attempt to load a .env file in project root (two levels up from this file)
        project_root = Path(__file__).resolve().parent.parent
        env_path = project_root / '.env'
        if env_path.exists():
            try:
                load_dotenv(env_path)  # silent load
                _DOTENV_LOADED = True
            except Exception as exc:  # pragma: no cover
                print(f"⚠️ Warning: failed to load .env file: {exc}")

    # Support legacy GOOGLE_API_KEY as fallback for Gemini
    gemini_raw = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY') or ''
    keys = {
        'openai': (os.environ.get('OPENAI_API_KEY') or '').strip(),
        'anthropic': (os.environ.get('ANTHROPIC_API_KEY') or '').strip(),
        'google': gemini_raw.strip(),  # Gemini
        'deepseek': (os.environ.get('DEEPSEEK_API_KEY') or '').strip(),
        'openrouter': (os.environ.get('OPENROUTER_API_KEY') or '').strip(),
    }

    # Placeholder detection
    def _is_placeholder(provider: str, value: str) -> bool:
        if not value or len(value) < 6:
            return True
        if provider == 'anthropic' and (value.startswith('sk-ant-xxx')):
            return True
        if provider == 'google' and value.startswith('AIzaxxxxxx'):
            return True
        if provider == 'deepseek' and value.startswith('sk-xxxxxxx'):
            return True
        if provider == 'openai' and not (value.startswith('sk-') or value.startswith('org-')):
            return True
        if provider == 'openrouter' and (
            'your-openrouter' in value.lower() or value.startswith('sk-or-placeholder')
        ):
            return True
        return False

    cleaned = {}
    for provider, raw in keys.items():
        if raw and not _is_placeholder(provider, raw):
            cleaned[provider] = raw
            print(f"✅ Found API key for {provider}: {raw[:5]}...")
        else:
            cleaned[provider] = None
            print(f"❌ No valid API key found for {provider}")

    return cleaned


def load_app_metadata():
    """Load optional application metadata used for API headers."""
    metadata = {
        'site_url': os.environ.get('SITE_URL'),
        'app_name': os.environ.get('APP_NAME')
    }

    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                app_config = config.get('app', {}) if isinstance(config, dict) else {}
                metadata['site_url'] = metadata['site_url'] or app_config.get('site_url')
                metadata['app_name'] = metadata['app_name'] or app_config.get('app_name')
        except Exception as exc:
            print(f"⚠️ Warning: failed to load app metadata from config.json: {exc}")

    if not metadata['site_url']:
        metadata['site_url'] = 'http://localhost:5005'

    if not metadata['app_name']:
        metadata['app_name'] = 'Obify Model Comparison Platform'

    return metadata

# Import centralized model configuration
from utils.model_config import (
    get_models_by_provider, calculate_cost
)


def clean_model_response(response_text):
    """Remove Markdown code fences (```json ... ``` ) from model responses."""
    if not isinstance(response_text, str):
        return response_text

    stripped = response_text.strip()
    if not stripped.startswith("```"):
        return response_text

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    cleaned = "\n".join(lines).strip()
    return cleaned if cleaned else response_text


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
    Count tokens accurately using tiktoken or provider-specific tokenizers.
    
    Args:
        text: The text to count tokens for
        model_name: The model to use for tokenization
        
    Returns:
        int: Number of tokens in the text
    """
    router_provider = None
    normalized_model = model_name

    if model_name.startswith('openrouter/'):
        parts = model_name.split('/')
        router_provider = parts[1] if len(parts) > 1 else None
        normalized_model = parts[-1]

        # Map routed model to underlying provider naming conventions
        if router_provider == 'openai':
            model_name = normalized_model
        elif router_provider in ('anthropic', 'claude'):
            model_name = normalized_model if normalized_model.startswith('claude-') else f"claude-{normalized_model}"
        elif router_provider == 'deepseek':
            model_name = normalized_model
        else:
            model_name = normalized_model
    else:
        normalized_model = model_name

    # Handle Anthropic/Claude models via their tokenizer when available
    if model_name.startswith('claude-'):
        try:
            from anthropic import Anthropic  # type: ignore
            client = Anthropic(api_key='placeholder')  # Real key supplied during API calls
            return client.count_tokens(text)
        except ImportError:
            print("Warning: anthropic library not available, using approximate token count")
            return len(text.split()) * 1.3
        except Exception as e:
            print(f"Error counting tokens for Claude model: {e}")
            return len(text.split()) * 1.3

    # DeepSeek (routed through OpenRouter) does not have a dedicated tokenizer in this project
    if router_provider == 'deepseek' or normalized_model.startswith('deepseek'):
        return len(text.split()) * 1.3

    import tiktoken

    try:
        if model_name.startswith('gpt-4'):
            encoding = tiktoken.encoding_for_model('gpt-4')
        elif model_name.startswith('gpt-3.5'):
            encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
        elif model_name.startswith('o1') or model_name.startswith('o3') or model_name.startswith('o4'):
            encoding = tiktoken.encoding_for_model('gpt-4')
        else:
            encoding = tiktoken.get_encoding('cl100k_base')

        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return len(text.split()) * 1.3

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
    
    start_time = time.time()
    
    try:
        # Retrieve Anthropic key strictly from environment loader
        api_key = load_api_keys().get('anthropic')
        if not api_key:
            return "Error: Anthropic API key not configured.", 0, 0, 0
        
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
        gemini_key = load_api_keys().get('google')
        if not gemini_key:
            return "Error: Gemini API key not configured.", 0, 0, 0
        
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

def call_openrouter_api(model_name, prompt, max_tokens=2048, temperature=0.7):
    """Invoke the OpenRouter chat completions endpoint."""
    import requests

    api_model_name = model_name
    if model_name.startswith('openrouter/'):
        api_model_name = model_name.split('openrouter/', 1)[1]

    print(f"🔄 OPENROUTER API CALL STARTED: {model_name} (API id: {api_model_name})")
    start_time = time.time()

    api_keys = load_api_keys()
    openrouter_key = api_keys.get('openrouter')

    if not openrouter_key:
        warning = "Error: OpenRouter API key not configured."
        print(f"❌ {warning}")
        return warning, 0, 0, 0, {}

    app_metadata = load_app_metadata()

    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "HTTP-Referer": app_metadata['site_url'],
        "X-Title": app_metadata['app_name'],
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes customer reviews."},
        {"role": "user", "content": prompt}
    ]

    payload = {
        "model": api_model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }

    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code >= 400:
                try:
                    error_body = response.json()
                    error_detail = error_body.get('error') if isinstance(error_body, dict) else None
                    error_message = error_detail.get('message') if isinstance(error_detail, dict) else response.text
                except Exception:
                    error_message = response.text

                print(f"❌ OpenRouter HTTP error ({response.status_code}): {error_message}")

                if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    print(f"   Retrying OpenRouter call ({attempt + 1}/{max_retries}) after {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue

                return f"Error: OpenRouter API returned {response.status_code}: {error_message}", 0, 0, time.time() - start_time, {}

            data = response.json()
            choices = data.get('choices', [])
            first_choice = choices[0] if choices else {}
            message_content = first_choice.get('message', {}).get('content', '') if isinstance(first_choice, dict) else ''

            usage = data.get('usage', {}) if isinstance(data, dict) else {}
            prompt_tokens = usage.get('prompt_tokens', 0) or 0
            completion_tokens = usage.get('completion_tokens', 0) or 0

            if prompt_tokens == 0:
                prompt_tokens = count_tokens(prompt, model_name)
            if completion_tokens == 0 and message_content:
                completion_tokens = count_tokens(message_content, model_name)

            latency = time.time() - start_time

            raw_total_cost = usage.get('total_cost') if isinstance(usage, dict) else None
            try:
                actual_cost = float(raw_total_cost) if raw_total_cost is not None else None
            except (TypeError, ValueError):
                actual_cost = None

            print(f"✅ OPENROUTER API CALL COMPLETED: {model_name}")
            print(f"⏱️ Time: {latency:.2f} seconds")
            print(f"🔤 Tokens: {prompt_tokens} prompt + {completion_tokens} completion = {prompt_tokens + completion_tokens} total")
            if actual_cost is not None:
                print(f"💲 Usage cost reported by OpenRouter: ${actual_cost:.6f}")

            if len(message_content) > 300:
                preview = message_content[:300] + "..."
            else:
                preview = message_content
            print(f"📝 Response preview: {preview}")

            metadata = {
                'actual_cost': actual_cost,
                'usage': usage
            }

            return message_content, prompt_tokens, completion_tokens, latency, metadata

        except requests.exceptions.Timeout:
            print(f"⏳ OpenRouter request timed out (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            return "Error: OpenRouter request timed out.", 0, 0, time.time() - start_time, {}
        except requests.exceptions.RequestException as exc:
            print(f"❌ OpenRouter request failed: {exc}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            return f"Error: OpenRouter request failed: {exc}", 0, 0, time.time() - start_time, {}
        except Exception as exc:
            print(f"❌ Unexpected error during OpenRouter call: {exc}")
            return f"Error: {exc}", 0, 0, time.time() - start_time, {}

    return "Error: OpenRouter retries exhausted.", 0, 0, time.time() - start_time, {}


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
        total_cost_value = 0.0
        
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
                call_metadata = {}

                # Route to the appropriate provider implementation
                cost_model_reference = model

                if model.startswith('openrouter/'):
                    call_result = call_openrouter_api(model, prompt)
                elif model.startswith('deepseek'):
                    routed_model = model
                    if not model.startswith('openrouter/'):
                        routed_model = f"openrouter/{model}"
                    cost_model_reference = routed_model
                    call_result = call_openrouter_api(routed_model, prompt)
                elif model.startswith('claude-'):
                    call_result = call_claude(model, prompt)
                elif model.startswith('gemini-'):
                    call_result = call_gemini(model, prompt)
                else:
                    call_result = call_openai(model, prompt)

                if isinstance(call_result, tuple) and len(call_result) == 5:
                    response, prompt_tokens, completion_tokens, latency, call_metadata = call_result
                elif isinstance(call_result, tuple) and len(call_result) == 4:
                    response, prompt_tokens, completion_tokens, latency = call_result
                else:
                    response = str(call_result)
                    prompt_tokens = 0
                    completion_tokens = 0
                    latency = 0

                response = clean_model_response(response)

                end_time = time.time()
                total_time = end_time - start_time

                # Check if response was successful
                success = not (isinstance(response, str) and response.startswith("Error:"))

                if success:
                    successful_calls += 1
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    total_latency += latency
                else:
                    errors += 1
                
                # Calculate cost using the centralized pricing calculation function
                cost_model_name = cost_model_reference.replace('openai:', '')
                total_cost = calculate_cost(cost_model_name, prompt_tokens, completion_tokens)

                actual_cost = call_metadata.get('actual_cost') if isinstance(call_metadata, dict) else None
                if isinstance(actual_cost, (int, float)):
                    total_cost = actual_cost

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

                if isinstance(call_metadata, dict):
                    if call_metadata.get('actual_cost') is not None:
                        result['actual_cost'] = call_metadata['actual_cost']
                    if call_metadata.get('usage') is not None:
                        result['usage'] = call_metadata['usage']
                
                model_results.append(result)

                status = "✅" if success else "❌"
                print(f"    {status} Review {i+1}: {total_time:.2f}s, {prompt_tokens+completion_tokens} tokens")

                total_cost_value += total_cost if isinstance(total_cost, (int, float)) else 0.0
                
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
            'average_latency': total_latency / successful_calls if successful_calls > 0 else 0,
            'total_cost': total_cost_value
        }
        
        # Print summary for this model
        print(f"  Summary for {model}: {successful_calls}/{len(reviews)} successful, {errors} errors")
    
    return results
