# OpenRouter Integration Plan (Updated)

## 1. Configuration & Secrets
- Extend `utils/model_handler.load_api_keys()` at line 19-63 to add:
  ```python
  'openrouter': os.environ.get('OPENROUTER_API_KEY'),
  ```
  Add placeholder detection for OpenRouter keys starting with 'sk-or-xxx' pattern.
- DeepSeek will remain a separate provider (already has key support at line 23) - not routing through OpenRouter.
- Surface OpenRouter key availability in `app.py` startup config (lines 100-139) following existing pattern for Claude/Gemini models.
- Update `config.template.json` to add `"openrouter": "your-openrouter-api-key-here"` entry.

## 2. Model Metadata & Pricing
- Extend `MODEL_PROVIDERS` dict (line 24-28) with:
  ```python
  'openrouter/': 'openrouter',
  'deepseek-': 'deepseek'  # Keep DeepSeek as separate provider
  ```
- Add OpenRouter entries to `MODEL_DEFINITIONS` (starting line 31):
  ```python
  'openrouter/openai/gpt-4o': {
      'provider': 'openrouter',
      'display_name': 'GPT-4o (via OpenRouter)',
      'description': 'OpenAI GPT-4o accessed through OpenRouter',
      'max_tokens': 32768,
      'default_max_output_tokens': 4096,
      'default_temperature': 0.7,
  },
  # Similar entries for other models
  ```
- Add to `MODEL_PRICING` (starting line 228) with OpenRouter's rates:
  ```python
  'openrouter/openai/gpt-4o': {'input': 5.00, 'output': 15.00},  # OpenRouter pricing
  ```
- Update `calculate_cost()` (lines 322-366) to check for API-provided `total_cost` first.

## 3. API Client Implementation
- Create `call_openrouter_api()` in `utils/model_handler.py` after line 441 (following Gemini pattern):
  ```python
  def call_openrouter_api(model_name, prompt, max_tokens=500, temperature=0.1):
      """Call OpenRouter API with the given prompt"""
      import requests
      import json
      import time

      start_time = time.time()

      # Load API key
      api_keys = load_api_keys()
      openrouter_key = api_keys.get('openrouter')

      if not openrouter_key:
          # Try config file fallback
          config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
          # ... (follow existing pattern from lines 339-346)

      headers = {
          "Authorization": f"Bearer {openrouter_key}",
          "HTTP-Referer": os.environ.get('SITE_URL', 'http://localhost:5000'),
          "X-Title": "Obify Model Comparison Platform",
          "Content-Type": "application/json"
      }

      # Strip 'openrouter/' prefix for API
      api_model = model_name.replace('openrouter/', '')

      payload = {
          "model": api_model,
          "messages": [{"role": "user", "content": prompt}],
          "max_tokens": max_tokens,
          "temperature": temperature,
          "stream": False
      }

      # Retry logic (similar to lines 214-274)
      max_retries = 3
      # ... implement retry with exponential backoff
  ```
- Update `test_models()` dispatcher (lines 564-570):
  ```python
  elif model.startswith('openrouter/'):
      response, prompt_tokens, completion_tokens, latency = call_openrouter_api(model, prompt)
  elif model.startswith('deepseek-'):
      # Separate DeepSeek implementation (future enhancement)
      response = f"Error: DeepSeek models not yet implemented", 0, 0, 0
  ```

## 4. Token Counting & Validation
- Update `count_tokens()` function (lines 110-161) to handle OpenRouter models:
  ```python
  # Add after line 139 (Claude handling)
  elif model_name.startswith('openrouter/'):
      # Strip prefix and use underlying model's tokenizer
      base_model = model_name.replace('openrouter/', '')
      if 'gpt' in base_model or 'openai' in base_model:
          # Use OpenAI tokenizer
          encoding = tiktoken.encoding_for_model('gpt-4')
          return len(encoding.encode(text))
      elif 'claude' in base_model:
          # Use Claude approximation
          return len(text.split()) * 1.3
      else:
          # Generic approximation
          return len(text.split()) * 1.3
  ```
- DeepSeek will use generic approximation (1.3 tokens per word) until specific tokenizer available.
- Skip validation for OpenRouter models (no equivalent to `validate_openai_model()`).

## 5. UI & Workflow Exposure
- Modify `get_available_models()` in `app.py` (lines 171-206) to add OpenRouter section:
  ```python
  # Add after Gemini models (around line 100)
  openrouter_models = [
      'openrouter/openai/gpt-4o',
      'openrouter/openai/gpt-4o-mini',
      'openrouter/anthropic/claude-3.5-sonnet',
      'openrouter/google/gemini-pro',
      'openrouter/meta/llama-3.1-70b-instruct',
  ]

  # In formatted_models section (around line 130)
  if 'openrouter' in api_keys and api_keys['openrouter']:
      formatted_models['openrouter'] = [
          {'id': model, 'name': model.split('/')[-1].replace('-', ' ').title(),
           'provider': 'OpenRouter Proxy'}
          for model in openrouter_models
      ]
  ```
- Templates already support dynamic provider groups via Jinja2 loops (line 141 in configure.html).
- Report generator already handles model name display; add mapping in `generate_report()` to clean display names.

## 6. Cost & Reporting Adjustments
- In `test_models()` result handling (lines 586-604), update cost calculation:
  ```python
  # After line 586
  if model.startswith('openrouter/') and hasattr(response, 'total_cost'):
      total_cost = response.total_cost  # Use API-provided cost
  else:
      total_cost = calculate_cost(model.replace('openai:', ''), prompt_tokens, completion_tokens)
  ```
- Update `report_generator.py` to handle slash-prefixed names:
  ```python
  # Add helper function for display names
  def get_display_name(model_id):
      if '/' in model_id:
          parts = model_id.split('/')
          return f"{parts[-1]} (via {parts[0].title()})"
      return model_id
  ```
- Evaluator already uses model IDs as keys, no changes needed.

## 7. Documentation & Samples
- Refresh `README.md`, `docs/CONTRIBUTING_MODELS.md`, and any quickstart docs with OpenRouter setup instructions, header requirements, and pricing notes.
- Provide example environment/config snippets including optional `SITE_URL`/`APP_NAME` values.

## 8. Testing & Validation

### 8.1 Unit Tests (create `tests/test_openrouter.py`)
```python
import unittest
from unittest.mock import patch, MagicMock
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_handler import load_api_keys, count_tokens, call_openrouter_api
from utils.model_config import (
    get_model_provider, get_models_by_provider,
    calculate_cost, get_model_pricing, MODEL_DEFINITIONS
)

class TestOpenRouterIntegration(unittest.TestCase):

    def test_api_key_loading(self):
        """Test OpenRouter API key loading from environment"""
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'sk-or-test-key'}):
            keys = load_api_keys()
            self.assertEqual(keys['openrouter'], 'sk-or-test-key')

    def test_placeholder_detection(self):
        """Test detection of placeholder OpenRouter keys"""
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'sk-or-xxx'}):
            keys = load_api_keys()
            self.assertIsNone(keys['openrouter'])  # Should detect as placeholder

    def test_model_provider_detection(self):
        """Test provider detection for OpenRouter models"""
        provider = get_model_provider('openrouter/openai/gpt-4o')
        self.assertEqual(provider, 'openrouter')

        provider = get_model_provider('openrouter/anthropic/claude-3.5-sonnet')
        self.assertEqual(provider, 'openrouter')

    def test_cost_calculation_standard(self):
        """Test cost calculation for OpenRouter models"""
        cost = calculate_cost('openrouter/openai/gpt-4o', 1000, 500)
        self.assertGreater(cost, 0)
        # Verify specific calculation: (1000/1M * 5.00) + (500/1M * 15.00)
        expected = (1000/1000000 * 5.00) + (500/1000000 * 15.00)
        self.assertAlmostEqual(cost, expected, places=6)

    def test_token_counting_openrouter(self):
        """Test token counting strips prefix correctly"""
        test_text = "This is a test prompt for token counting"

        # Test OpenRouter/OpenAI model
        tokens = count_tokens(test_text, "openrouter/openai/gpt-4o")
        self.assertGreater(tokens, 0)

        # Test OpenRouter/Claude model
        tokens = count_tokens(test_text, "openrouter/anthropic/claude-3.5-sonnet")
        self.assertGreater(tokens, 0)

        # Test generic OpenRouter model
        tokens = count_tokens(test_text, "openrouter/meta/llama-3.1-70b")
        self.assertGreater(tokens, 0)

    def test_models_by_provider(self):
        """Test getting all OpenRouter models"""
        # Add test models to MODEL_DEFINITIONS
        MODEL_DEFINITIONS['openrouter/openai/test'] = {
            'provider': 'openrouter',
            'display_name': 'Test Model'
        }

        openrouter_models = get_models_by_provider('openrouter')
        self.assertIn('openrouter/openai/test', openrouter_models)
```

### 8.2 API Client Tests with Mocking
```python
class TestOpenRouterAPI(unittest.TestCase):

    @patch('requests.post')
    def test_successful_api_call(self, mock_post):
        """Test successful OpenRouter API call"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Test response'}}],
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 20,
                'total_cost': 0.0003
            }
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key'}):
            response, prompt_tokens, completion_tokens, latency = call_openrouter_api(
                'openrouter/openai/gpt-4o', 'Test prompt'
            )

        self.assertEqual(response, 'Test response')
        self.assertEqual(prompt_tokens, 10)
        self.assertEqual(completion_tokens, 20)
        self.assertGreater(latency, 0)

    @patch('requests.post')
    def test_api_retry_on_429(self, mock_post):
        """Test retry logic on rate limit (429) errors"""
        # First call fails with 429, second succeeds
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.raise_for_status.side_effect = requests.exceptions.HTTPError()

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            'choices': [{'message': {'content': 'Success after retry'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 20}
        }

        mock_post.side_effect = [mock_response_429, mock_response_success]

        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key'}):
            response, _, _, _ = call_openrouter_api(
                'openrouter/openai/gpt-4o', 'Test prompt'
            )

        self.assertEqual(response, 'Success after retry')
        self.assertEqual(mock_post.call_count, 2)  # Verify retry happened

    @patch('requests.post')
    def test_api_error_handling(self, mock_post):
        """Test error message handling from API"""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            'error': {
                'code': 'invalid_request',
                'message': 'Invalid model specified'
            }
        }
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key'}):
            response, _, _, _ = call_openrouter_api(
                'openrouter/invalid/model', 'Test prompt'
            )

        self.assertIn('Error:', response)
        self.assertIn('Invalid model', response)

    def test_missing_api_key(self):
        """Test behavior when API key is missing"""
        with patch.dict(os.environ, {}, clear=True):
            response, _, _, _ = call_openrouter_api(
                'openrouter/openai/gpt-4o', 'Test prompt'
            )

        self.assertIn('Error:', response)
        self.assertIn('API key not found', response.lower())
```

### 8.3 Integration Tests
```python
class TestOpenRouterIntegration(unittest.TestCase):

    def test_model_in_test_models_dispatcher(self):
        """Test OpenRouter models work in test_models dispatcher"""
        from utils.model_handler import test_models

        reviews = ["Great product!"]
        models = ["openrouter/openai/gpt-4o"]
        prompt_template = "Summarize: {review}"

        with patch('utils.model_handler.call_openrouter_api') as mock_api:
            mock_api.return_value = ("Summary", 10, 5, 0.1)

            results = test_models(reviews, models, prompt_template)

            self.assertIn('openrouter/openai/gpt-4o', results)
            self.assertEqual(results['openrouter/openai/gpt-4o'][0]['response'], "Summary")

    def test_cost_preference_total_cost(self):
        """Test that API-provided total_cost is preferred over calculated"""
        # Simulate response with total_cost
        mock_response = type('Response', (), {
            'total_cost': 0.0005  # API-provided cost
        })

        calculated_cost = calculate_cost('openrouter/openai/gpt-4o', 1000, 500)

        # In actual implementation, should prefer mock_response.total_cost
        self.assertNotEqual(calculated_cost, 0.0005)  # They should differ
```

### 8.4 End-to-End Test Scenarios
```python
class TestOpenRouterE2E(unittest.TestCase):

    def test_full_workflow(self):
        """Test complete workflow from model selection to report generation"""
        # This would be run manually or in CI with actual API key

        test_plan = """
        1. Set OPENROUTER_API_KEY in environment
        2. Start Flask app
        3. Verify /api/models includes OpenRouter models
        4. Upload test CSV with sample reviews
        5. Select OpenRouter models via UI
        6. Submit test run
        7. Verify response includes OpenRouter results
        8. Download PDF report
        9. Verify report contains OpenRouter model results with costs
        """

        # Automated version with mocking
        from app import app, get_available_models

        with app.test_client() as client:
            with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key'}):
                # Test model availability endpoint
                models = get_available_models()
                self.assertIn('openrouter', models)

                # Test run endpoint accepts OpenRouter models
                response = client.post('/run_test', json={
                    'models': ['openrouter/openai/gpt-4o'],
                    'reviews': ['Test review'],
                    'prompt_template': 'Summarize: {review}'
                })
                # Would need proper mocking of the full flow
```

### 8.5 Manual QA Checklist
1. **Environment Setup**
   - [ ] Set `OPENROUTER_API_KEY` in config.json or environment
   - [ ] Verify key is not a placeholder (doesn't start with 'sk-or-xxx')
   - [ ] Set optional `SITE_URL` environment variable

2. **Application Startup**
   - [ ] Run `python app.py`
   - [ ] Check console logs for "Found API key for openrouter"
   - [ ] Verify no error messages about OpenRouter key

3. **UI Verification**
   - [ ] Navigate to http://localhost:5000
   - [ ] Click "Configure Test"
   - [ ] Verify "OpenRouter" section appears with models
   - [ ] Check tooltips explain proxy behavior

4. **Test Execution**
   - [ ] Upload sample reviews CSV
   - [ ] Select at least 2 OpenRouter models + 1 direct provider model
   - [ ] Run test
   - [ ] Monitor console for API calls to OpenRouter
   - [ ] Verify no errors in response

5. **Results Validation**
   - [ ] Check all selected models have responses
   - [ ] Verify token counts are present
   - [ ] Confirm costs are calculated (or API-provided if available)
   - [ ] Compare OpenRouter vs direct provider responses

6. **Report Generation**
   - [ ] Download PDF report
   - [ ] Verify OpenRouter models appear with proper names
   - [ ] Check cost comparison chart includes OpenRouter
   - [ ] Validate performance metrics are calculated

7. **Error Scenarios**
   - [ ] Test with invalid API key - should show clear error
   - [ ] Test with rate limiting (many rapid requests)
   - [ ] Test with unsupported model ID
   - [ ] Verify graceful degradation

## 9. Future Enhancements (Optional)
- Investigate OpenRouter batch endpoints for multi-review optimization once baseline integration is stable.
- Consider caching `/models` response to auto-populate model lists dynamically.
