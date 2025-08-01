import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_openai_key(openai_key):
    """
    A clean implementation to validate an OpenAI API key
    
    Args:
        openai_key: The OpenAI API key to validate
        
    Returns:
        tuple: (is_valid, error_message, available_models)
    """
    if not openai_key:
        return False, "No API key provided", []
        
    # Check basic format
    if not (openai_key.startswith('sk-') or openai_key.startswith('org-')):
        return False, f"Invalid API key format. Key should start with 'sk-' not '{openai_key[:5]}'", []
    
    # Clear any proxy settings that might interfere
    os.environ.pop('http_proxy', None)
    os.environ.pop('https_proxy', None)
    
    try:
        # Import here to avoid any potential module-level issues
        import openai
        
        # Set the API key at the module level for older versions
        openai.api_key = openai_key
        
        # Try to list models using the simplest approach possible
        try:
            # First try with the v1+ approach, but without creating a client
            # This avoids the proxies parameter issue completely
            start_time = time.time()
            logger.info("Testing API connection using v1+ module-level approach...")
            models = openai.models.list()
            model_data = getattr(models, 'data', models)
            available_models = [getattr(model, 'id', model) for model in model_data]
            logger.info(f"✅ API connection successful! Found {len(available_models)} models in {time.time() - start_time:.2f}s")
            return True, None, available_models
        except Exception as e1:
            logger.warning(f"Modern API approach failed: {str(e1)}")
            
            # Fall back to legacy approach
            try:
                logger.info("Testing API connection using legacy approach...")
                models = openai.Model.list()
                if isinstance(models, dict) and 'data' in models:
                    available_models = [m.get('id', 'unknown') for m in models['data']]
                else:
                    available_models = [getattr(m, 'id', str(m)) for m in models]
                logger.info(f"✅ API connection successful (legacy)! Found {len(available_models)} models")
                return True, None, available_models
            except Exception as e2:
                logger.error(f"Legacy approach also failed: {str(e2)}")
                error_msg = f"Failed to validate OpenAI API key: {str(e2)}"
                return False, error_msg, []
                
    except Exception as e:
        error_msg = f"Error during OpenAI API key validation: {str(e)}"
        logger.error(error_msg)
        return False, error_msg, []


def get_openai_client(api_key):
    """
    Get an OpenAI client with proper error handling
    
    Args:
        api_key: OpenAI API key
        
    Returns:
        tuple: (client, is_legacy)
    """
    # Clear any proxy settings
    os.environ.pop('http_proxy', None)
    os.environ.pop('https_proxy', None)
    
    try:
        import openai
        try:
            # Try modern client (v1+)
            client = openai.OpenAI(api_key=api_key)
            return client, False
        except Exception:
            # Fall back to legacy approach
            openai.api_key = api_key
            return openai, True
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        raise
