# Contributing to Obify: Adding or Updating Models

This guide explains how to add new models or update existing models in the Obify platform.

## Table of Contents
1. [Overview](#overview)
2. [Model Configuration File](#model-configuration-file)
3. [Adding a New Model](#adding-a-new-model)
4. [Updating an Existing Model](#updating-an-existing-model)
5. [Adding a New Provider](#adding-a-new-provider)
6. [Testing Your Changes](#testing-your-changes)

## Overview

Obify uses a centralized configuration approach for all model definitions, pricing, and calculations. All model-related settings are stored in a single file: `utils/model_config.py`. This makes it easy to add new models or update existing ones without having to modify multiple parts of the codebase.

## Model Configuration File

The `utils/model_config.py` file contains several key components:

1. **MODEL_PROVIDERS**: Maps model prefixes (e.g., 'gpt-', 'claude-') to their providers (OpenAI, Anthropic, Google)
2. **MODEL_DEFINITIONS**: Contains detailed information about each model, including:
   - `provider`: The API provider (openai, anthropic, google)
   - `display_name`: Human-readable name for the UI
   - `description`: Brief description of the model
   - `max_tokens`: Maximum context window size
   - `default_max_output_tokens`: Default generation limit
   - `default_temperature`: Default temperature setting
   - `has_tiered_pricing`: (Optional) Whether the model uses tiered pricing
   - `tier_threshold`: (Optional) Threshold for tier 2 pricing

3. **MODEL_PRICING**: Contains pricing information for each model:
   - `input`: Cost per 1M input tokens (in USD)
   - `output`: Cost per 1M output tokens (in USD)
   - For tiered pricing models:
     - `tier_threshold`: Threshold for tier 2 pricing
     - `tier2_input`: Tier 2 input cost
     - `tier2_output`: Tier 2 output cost

4. **Helper functions**:
   - `calculate_cost()`: Calculates the cost based on model and token usage
   - `get_model_provider()`: Determines which provider a model belongs to
   - `get_models_by_provider()`: Gets all models for a specific provider

## Adding a New Model

To add a new model to Obify:

1. Determine if the model is from an existing provider (OpenAI, Anthropic, Google) or a new provider.

2. Add the model to `MODEL_DEFINITIONS` with all required parameters:

```python
# Example of adding a new OpenAI model
MODEL_DEFINITIONS['gpt-5'] = {
    'provider': 'openai',
    'display_name': 'GPT-5',
    'description': 'Next generation large language model',
    'max_tokens': 128000,  # 128k context window
    'default_max_output_tokens': 4096,
    'default_temperature': 0.7,
}
```

3. Add pricing information to `MODEL_PRICING`:

```python
# Standard pricing model
MODEL_PRICING['gpt-5'] = {
    'input': 10.00,   # $10 per 1M input tokens
    'output': 30.00,  # $30 per 1M output tokens
}

# OR for tiered pricing
MODEL_PRICING['gpt-5'] = {
    'input': 10.00,
    'output': 30.00,
    'tier_threshold': 1000000,  # 1M tokens before tier 2
    'tier2_input': 5.00,        # $5 per 1M input tokens after threshold
    'tier2_output': 15.00       # $15 per 1M output tokens after threshold
}
```

4. If your model has specific versioned variants, add them to the `add_versioned_models()` function:

```python
# Inside add_versioned_models()
versioned_models['gpt-5-20260401'] = {**MODEL_DEFINITIONS['gpt-5']}
```

## Updating an Existing Model

To update an existing model (e.g., changing the price):

1. Find the model in `MODEL_PRICING` and update its values:

```python
# Update pricing for GPT-4o
MODEL_PRICING['gpt-4o'] = {
    'input': 4.00,   # Changed from $5.00 to $4.00
    'output': 16.00  # Changed from $20.00 to $16.00
}
```

2. If needed, update the model's details in `MODEL_DEFINITIONS`:

```python
# Update GPT-4o context window
MODEL_DEFINITIONS['gpt-4o']['max_tokens'] = 65536  # Increasing from 32k to 64k
```

## Adding a New Provider

To add support for a completely new provider:

1. Add the provider prefix to `MODEL_PROVIDERS`:

```python
MODEL_PROVIDERS['llama-'] = 'meta'
```

2. Add your models to `MODEL_DEFINITIONS` with the new provider:

```python
MODEL_DEFINITIONS['llama-3-70b'] = {
    'provider': 'meta',
    'display_name': 'Llama 3 (70B)',
    'description': 'Meta\'s flagship 70B parameter model',
    'max_tokens': 16384,
    'default_max_output_tokens': 2048,
    'default_temperature': 0.7,
}
```

3. Add pricing information to `MODEL_PRICING`:

```python
MODEL_PRICING['llama-3-70b'] = {
    'input': 1.00,
    'output': 3.00,
}
```

4. Implement the API integration in `model_handler.py` by:
   - Adding a new function to call the provider's API (similar to `call_openai`, `call_anthropic_api`, etc.)
   - Adding a new wrapper function (similar to `call_openai`, `call_claude`, etc.)
   - Updating the model calling logic in `test_models`

## Testing Your Changes

After adding or updating models:

1. Run the application: `python app.py`
2. Try selecting your new model in the configure page
3. Run a test with a small sample to verify:
   - The model appears correctly in the UI
   - The API calls work correctly (if you've added a new provider)
   - The pricing calculation is correct in the reports

For pricing calculation testing, you can also use the functions directly:

```python
from utils.model_config import calculate_cost

# Test with some token values
cost = calculate_cost('your-new-model', 1000, 500)
print(f"Cost for processing: ${cost:.6f}")
```

## Commit Guidelines

When committing changes to models:

1. Use clear commit messages (e.g., "Add support for Model X" or "Update pricing for Model Y")
2. Include any relevant API documentation links in your commit message
3. If you've changed pricing, note the date of the pricing update in your commit

Thank you for contributing to Obify!
