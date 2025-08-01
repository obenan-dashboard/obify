"""
Model Configuration Module for Obify

This module centralizes all model definitions, pricing information, and formulas used throughout the Obify application.
This makes it easy for contributors to add new models or update pricing without having to modify multiple files.

Structure:
- MODEL_PROVIDERS: Dictionary mapping model prefixes to their providers (OpenAI, Anthropic, Google)
- MODEL_DEFINITIONS: Complete list of supported models with their details
- MODEL_PRICING: Pricing information for all supported models
- calculate_cost: Function to calculate cost based on model, tokens, and pricing structure

Usage:
    from utils.model_config import MODEL_DEFINITIONS, MODEL_PRICING, calculate_cost
    
    # Get list of all OpenAI models
    openai_models = [model for model, info in MODEL_DEFINITIONS.items() if info['provider'] == 'openai']
    
    # Calculate cost for a specific model
    cost = calculate_cost('gpt-4o', prompt_tokens, completion_tokens)
"""

# Model provider mapping - used to determine which API to call for a given model
MODEL_PROVIDERS = {
    'gpt-': 'openai',
    'claude-': 'anthropic',
    'gemini-': 'google'
}

# Complete list of all supported models with their details
MODEL_DEFINITIONS = {
    # OpenAI models
    'gpt-3.5-turbo': {
        'provider': 'openai',
        'display_name': 'GPT-3.5 Turbo',
        'description': 'Good balance of capabilities and cost',
        'max_tokens': 16385,  # 16k context window
        'default_max_output_tokens': 2048,
        'default_temperature': 0.7,
    },
    'gpt-4': {
        'provider': 'openai',
        'display_name': 'GPT-4',
        'description': 'More capable but slower and more expensive',
        'max_tokens': 8192,  # 8k context window
        'default_max_output_tokens': 2048,
        'default_temperature': 0.7,
    },
    'gpt-4o': {
        'provider': 'openai',
        'display_name': 'GPT-4o',
        'description': 'GPT-4 Omni - latest multimodal model',
        'max_tokens': 32768,  # 32k context window
        'default_max_output_tokens': 4096,
        'default_temperature': 0.7,
    },
    'gpt-4o-mini': {
        'provider': 'openai',
        'display_name': 'GPT-4o Mini',
        'description': 'Smaller, faster version of GPT-4o',
        'max_tokens': 16385,
        'default_max_output_tokens': 2048,
        'default_temperature': 0.7,
    },
    'gpt-4.1': {
        'provider': 'openai',
        'display_name': 'GPT-4.1',
        'description': 'Improved version of GPT-4',
        'max_tokens': 32768,
        'default_max_output_tokens': 4096,
        'default_temperature': 0.7,
    },
    'gpt-4.1-mini': {
        'provider': 'openai',
        'display_name': 'GPT-4.1 Mini',
        'description': 'Smaller, faster version of GPT-4.1',
        'max_tokens': 16385,
        'default_max_output_tokens': 2048,
        'default_temperature': 0.7,
    },
    'gpt-4.1-nano': {
        'provider': 'openai',
        'display_name': 'GPT-4.1 Nano',
        'description': 'Smallest, fastest version of GPT-4.1',
        'max_tokens': 8192,
        'default_max_output_tokens': 1024,
        'default_temperature': 0.7,
    },
    
    # Anthropic Claude models
    'claude-3-7-sonnet': {
        'provider': 'anthropic',
        'display_name': 'Claude 3.7 Sonnet',
        'description': 'Anthropic\'s mid-tier Claude 3.7 model',
        'max_tokens': 200000,
        'default_max_output_tokens': 4096,
        'default_temperature': 0.5,
    },
    'claude-3-5-sonnet': {
        'provider': 'anthropic',
        'display_name': 'Claude 3.5 Sonnet',
        'description': 'Anthropic\'s mid-tier Claude 3.5 model',
        'max_tokens': 200000,
        'default_max_output_tokens': 4096,
        'default_temperature': 0.5,
    },
    'claude-3-5-haiku': {
        'provider': 'anthropic',
        'display_name': 'Claude 3.5 Haiku',
        'description': 'Anthropic\'s fastest, most efficient Claude 3.5 model',
        'max_tokens': 200000,
        'default_max_output_tokens': 4096,
        'default_temperature': 0.5,
    },
    'claude-3-haiku': {
        'provider': 'anthropic',
        'display_name': 'Claude 3 Haiku',
        'description': 'Anthropic\'s fastest, most efficient Claude 3 model',
        'max_tokens': 200000,
        'default_max_output_tokens': 4096,
        'default_temperature': 0.5,
    },
    'claude-3-opus': {
        'provider': 'anthropic',
        'display_name': 'Claude 3 Opus',
        'description': 'Anthropic\'s most powerful Claude 3 model',
        'max_tokens': 200000,
        'default_max_output_tokens': 4096,
        'default_temperature': 0.5,
    },
    'claude-2.1': {
        'provider': 'anthropic',
        'display_name': 'Claude 2.1',
        'description': 'Anthropic\'s Claude 2.1 model',
        'max_tokens': 100000,
        'default_max_output_tokens': 4096,
        'default_temperature': 0.5,
    },
    'claude-2.0': {
        'provider': 'anthropic',
        'display_name': 'Claude 2.0',
        'description': 'Anthropic\'s Claude 2.0 model',
        'max_tokens': 100000,
        'default_max_output_tokens': 4096,
        'default_temperature': 0.5,
    },
    
    # Google Gemini models
    'gemini-2.5-pro': {
        'provider': 'google',
        'display_name': 'Gemini 2.5 Pro',
        'description': 'Google\'s most capable Gemini model with tiered pricing',
        'max_tokens': 1048576,  # 1M context window
        'default_max_output_tokens': 8192,
        'default_temperature': 0.7,
        'has_tiered_pricing': True,
        'tier_threshold': 200000,  # 200k tokens before tier 2 pricing kicks in
    },
    'gemini-2.5-flash': {
        'provider': 'google',
        'display_name': 'Gemini 2.5 Flash',
        'description': 'Google\'s faster Gemini model',
        'max_tokens': 1048576,  # 1M context window
        'default_max_output_tokens': 8192,
        'default_temperature': 0.7,
    },
    'gemini-2.5-flash-lite': {
        'provider': 'google',
        'display_name': 'Gemini 2.5 Flash Lite',
        'description': 'Google\'s smaller, more efficient Gemini model',
        'max_tokens': 1048576,  # 1M context window
        'default_max_output_tokens': 8192,
        'default_temperature': 0.7,
    },
    'gemini-2.0-flash': {
        'provider': 'google',
        'display_name': 'Gemini 2.0 Flash',
        'description': 'Google\'s Gemini 2.0 model',
        'max_tokens': 524288,  # 512k context window
        'default_max_output_tokens': 8192,
        'default_temperature': 0.7,
    },
    'gemini-2.0-flash-lite': {
        'provider': 'google',
        'display_name': 'Gemini 2.0 Flash Lite',
        'description': 'Google\'s smaller Gemini 2.0 model',
        'max_tokens': 524288,  # 512k context window
        'default_max_output_tokens': 8192,
        'default_temperature': 0.7,
    }
}

# Helper function to add versioned model variants
def add_versioned_models():
    """Add versioned variants of base models to MODEL_DEFINITIONS"""
    versioned_models = {
        # OpenAI versioned models
        'gpt-4-0613': {**MODEL_DEFINITIONS['gpt-4']},
        'gpt-4-32k-0613': {**MODEL_DEFINITIONS['gpt-4'], 'max_tokens': 32768},
        'gpt-4-1106-preview': {**MODEL_DEFINITIONS['gpt-4'], 'display_name': 'GPT-4 Turbo (Preview)'},
        'gpt-4-vision-preview': {**MODEL_DEFINITIONS['gpt-4'], 'display_name': 'GPT-4 Vision (Preview)'},
        'gpt-3.5-turbo-0613': {**MODEL_DEFINITIONS['gpt-3.5-turbo']},
        'gpt-3.5-turbo-16k-0613': {**MODEL_DEFINITIONS['gpt-3.5-turbo'], 'max_tokens': 16385},
        'gpt-3.5-turbo-1106': {**MODEL_DEFINITIONS['gpt-3.5-turbo']},
        'gpt-4.1-2025-04-14': {**MODEL_DEFINITIONS['gpt-4.1']},
        'gpt-4o-2024-08-06': {**MODEL_DEFINITIONS['gpt-4o']},
        'gpt-4o-mini-2024-07-18': {**MODEL_DEFINITIONS['gpt-4o-mini']},
        
        # Claude versioned models
        'claude-3-7-sonnet-20250219': {**MODEL_DEFINITIONS['claude-3-7-sonnet']},
        'claude-3-5-sonnet-20241022': {**MODEL_DEFINITIONS['claude-3-5-sonnet']},
        'claude-3-5-sonnet-20240620': {**MODEL_DEFINITIONS['claude-3-5-sonnet']},
        'claude-3-sonnet-20240229': {**MODEL_DEFINITIONS['claude-3-5-sonnet'], 'display_name': 'Claude 3 Sonnet'},
        'claude-3-5-haiku-20241022': {**MODEL_DEFINITIONS['claude-3-5-haiku']},
        'claude-3-haiku-20240307': {**MODEL_DEFINITIONS['claude-3-haiku']},
        'claude-3-opus-20240229': {**MODEL_DEFINITIONS['claude-3-opus']},
    }
    
    # Add versioned models to the main dictionary
    for model_id, model_info in versioned_models.items():
        MODEL_DEFINITIONS[model_id] = model_info

# Call the function to populate versioned models
add_versioned_models()

# Model pricing information (cost per 1M tokens)
# All prices are in USD per 1M tokens as specified in the requirements
MODEL_PRICING = {
    # OpenAI Standard models
    'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},  # $0.50/1M input, $1.50/1M output
    'gpt-4': {'input': 30.00, 'output': 60.00},  # $30.00/1M input, $60.00/1M output
    'gpt-4-0613': {'input': 30.00, 'output': 60.00},  # $30.00/1M input, $60.00/1M output
    
    # GPT-4o models
    'gpt-4o': {'input': 5.00, 'output': 20.00},  # $5.00/1M input, $20.00/1M output
    'gpt-4o-mini': {'input': 0.15, 'output': 0.60},  # $0.15/1M input, $0.60/1M output
    
    # GPT-4.1 models
    'gpt-4.1': {'input': 2.00, 'output': 8.00},  # $2.00/1M input, $8.00/1M output
    'gpt-4.1-mini': {'input': 0.40, 'output': 1.60},  # $0.40/1M input, $1.60/1M output
    'gpt-4.1-nano': {'input': 0.10, 'output': 0.40},  # $0.10/1M input, $0.40/1M output
    
    # Include more specific versions for matching (with prices per 1M tokens)
    'gpt-4.1-2025-04-14': {'input': 2.00, 'output': 8.00},
    'gpt-4o-2024-08-06': {'input': 5.00, 'output': 20.00},
    'gpt-4o-mini-2024-07-18': {'input': 0.15, 'output': 0.60},
    
    # Claude models - Sonnet tier
    'claude-3-7-sonnet': {'input': 3.00, 'output': 15.00},  # $3/1M input, $15/1M output
    'claude-3-5-sonnet': {'input': 3.00, 'output': 15.00},  # $3/1M input, $15/1M output
    'claude-3-5-sonnet-20240620': {'input': 3.00, 'output': 15.00},  # $3/1M input, $15/1M output
    'claude-3-sonnet-20240229': {'input': 3.00, 'output': 15.00},  # $3/1M input, $15/1M output
    
    # Claude models - Haiku tier
    'claude-3-5-haiku': {'input': 0.80, 'output': 4.00},  # $0.80/1M input, $4/1M output
    'claude-3-haiku': {'input': 0.25, 'output': 1.25},  # $0.25/1M input, $1.25/1M output
    
    # Claude models - Opus tier
    'claude-3-opus': {'input': 15.00, 'output': 75.00},  # $15/1M input, $75/1M output
    
    # Claude models - Claude 2 series
    'claude-2.1': {'input': 8.00, 'output': 24.00},  # $8/1M input, $24/1M output
    'claude-2.0': {'input': 8.00, 'output': 24.00},  # $8/1M input, $24/1M output
    
    # Gemini models
    'gemini-2.5-pro': {
        'input': 7.00, 'output': 21.00,  # Tier 1 pricing (first 200k tokens)
        'tier_threshold': 200000,  # 200k tokens threshold before tier 2 pricing
        'tier2_input': 2.00, 'tier2_output': 6.00  # Tier 2 pricing (after 200k tokens)
    },
    'gemini-2.5-flash': {'input': 0.70, 'output': 2.10},  # $0.70/1M input, $2.10/1M output
    'gemini-2.5-flash-lite': {'input': 0.35, 'output': 1.05},  # $0.35/1M input, $1.05/1M output
    'gemini-2.0-flash': {'input': 0.35, 'output': 1.05},  # $0.35/1M input, $1.05/1M output
    'gemini-2.0-flash-lite': {'input': 0.175, 'output': 0.525}  # $0.175/1M input, $0.525/1M output
}

def get_model_provider(model_name):
    """
    Determine the provider for a given model name
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        str: Provider name ('openai', 'anthropic', 'google') or None if not found
    """
    if model_name in MODEL_DEFINITIONS:
        return MODEL_DEFINITIONS[model_name]['provider']
        
    # Check by prefix if not directly in MODEL_DEFINITIONS
    for prefix, provider in MODEL_PROVIDERS.items():
        if model_name.startswith(prefix):
            return provider
    
    return None

def get_model_pricing(model_name):
    """
    Get pricing information for a model
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        dict: Pricing information or default pricing if model not found
    """
    # Direct match
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]
    
    # Try to match by prefix for versioned models
    base_model = model_name.split('-')[0:2]  # Get first two parts of model name
    base_model = '-'.join(base_model)
    
    for key in MODEL_PRICING:
        if model_name.startswith(key) or key.startswith(base_model):
            return MODEL_PRICING[key]
    
    # Default fallback
    return MODEL_PRICING['gpt-3.5-turbo']

def calculate_cost(model_name, prompt_tokens, completion_tokens):
    """
    Calculate the cost for a model based on token usage
    
    Args:
        model_name (str): Name of the model
        prompt_tokens (int): Number of tokens in the prompt
        completion_tokens (int): Number of tokens in the completion
        
    Returns:
        float: Total cost in USD
    """
    # Get pricing for the model
    model_cost = get_model_pricing(model_name)
    
    # Handle tiered pricing models (like Gemini 2.5 Pro)
    if 'tier_threshold' in model_cost:
        tier_threshold = model_cost['tier_threshold']
        
        # Calculate input cost with tiered pricing
        if prompt_tokens <= tier_threshold:
            # All tokens at tier 1 rate
            input_cost = (prompt_tokens / 1000000) * model_cost['input']
        else:
            # First tier_threshold tokens at tier 1 rate, remaining at tier 2 rate
            tier1_input_cost = (tier_threshold / 1000000) * model_cost['input']
            tier2_input_cost = ((prompt_tokens - tier_threshold) / 1000000) * model_cost['tier2_input']
            input_cost = tier1_input_cost + tier2_input_cost
        
        # Calculate output cost with tiered pricing
        if completion_tokens <= tier_threshold:
            # All tokens at tier 1 rate
            output_cost = (completion_tokens / 1000000) * model_cost['output']
        else:
            # First tier_threshold tokens at tier 1 rate, remaining at tier 2 rate
            tier1_output_cost = (tier_threshold / 1000000) * model_cost['output']
            tier2_output_cost = ((completion_tokens - tier_threshold) / 1000000) * model_cost['tier2_output']
            output_cost = tier1_output_cost + tier2_output_cost
    else:
        # Standard pricing for all other models
        input_cost = (prompt_tokens / 1000000) * model_cost['input']
        output_cost = (completion_tokens / 1000000) * model_cost['output']
    
    total_cost = input_cost + output_cost
    return total_cost

def get_models_by_provider(provider):
    """
    Get all models for a specific provider
    
    Args:
        provider (str): Provider name ('openai', 'anthropic', 'google')
        
    Returns:
        list: List of model names for the specified provider
    """
    return [model for model, info in MODEL_DEFINITIONS.items() 
            if info['provider'] == provider and '-' in model]  # Filter out base models

def get_valid_models():
    """
    Get a list of all valid models
    
    Returns:
        list: List of all valid model names
    """
    return list(MODEL_DEFINITIONS.keys())

# Example usage in code comments:
"""
from utils.model_config import calculate_cost, get_models_by_provider

# Get all OpenAI models
openai_models = get_models_by_provider('openai')

# Calculate cost for a specific API call
cost = calculate_cost('gpt-4o', 1500, 500)  # 1500 prompt tokens, 500 completion tokens
"""
