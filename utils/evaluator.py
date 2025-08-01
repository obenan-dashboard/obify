import json
import re
import logging
from difflib import SequenceMatcher

# Set up logger
logger = logging.getLogger(__name__)

def is_valid_json(text):
    """
    Check if a string is valid JSON
    
    Args:
        text: String to check
        
    Returns:
        bool: True if valid JSON, False otherwise
    """
    try:
        # Try to extract JSON from the text
        json_pattern = r'\{[\s\S]*\}'
        match = re.search(json_pattern, text)
        if match:
            text = match.group(0)
        
        json.loads(text)
        return True
    except:
        return False

def extract_json_from_text(text):
    """
    Extract JSON from text response
    
    Args:
        text: Text containing JSON
        
    Returns:
        dict: Extracted JSON or None if not found
    """
    if not text:
        logger.warning(f"⚠️ Attempted to extract JSON from empty text")
        return None
    
    logger.info(f"🔍 Attempting to extract JSON from text: {text[:100]}...")
    
    try:
        # Try to find JSON pattern in text
        json_pattern = r'\{[\s\S]*\}'
        match = re.search(json_pattern, text)
        if match:
            json_str = match.group(0)
            logger.info(f"✅ Found JSON pattern in text, extracted: {json_str[:100]}...")
            try:
                parsed_json = json.loads(json_str)
                logger.info(f"✅ Successfully parsed JSON, keys: {list(parsed_json.keys()) if isinstance(parsed_json, dict) else 'Not a dict'}")
                return parsed_json
            except json.JSONDecodeError as e:
                logger.error(f"❌ Error parsing extracted JSON: {str(e)}")
        
        # If no JSON pattern found, try to parse the whole text
        logger.info(f"🔍 No JSON pattern found, trying to parse entire text as JSON")
        parsed_json = json.loads(text)
        logger.info(f"✅ Successfully parsed entire text as JSON, keys: {list(parsed_json.keys()) if isinstance(parsed_json, dict) else 'Not a dict'}")
        return parsed_json
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSONDecodeError: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error extracting JSON: {str(e)}")
        return None

def compare_json_structure(actual, expected_structure):
    """
    Compare if actual JSON follows the expected structure
    
    Args:
        actual: Actual JSON response
        expected_structure: Expected JSON structure
        
    Returns:
        float: Score between 0-1 representing structural accuracy
    """
    if not actual:
        return 0.0
    
    score = 0.0
    total_expected_keys = len(expected_structure.keys())
    found_keys = 0
    
    for key, expected_value in expected_structure.items():
        if key in actual:
            found_keys += 1
            if isinstance(expected_value, dict) and isinstance(actual[key], dict):
                subscore = compare_json_structure(actual[key], expected_value)
                score += subscore
    
    # Calculate structure score
    structure_score = found_keys / total_expected_keys if total_expected_keys > 0 else 0
    return structure_score

def evaluate_consistency(model_results):
    """
    Evaluate the consistency of the model responses across reviews
    
    Args:
        model_results: List of results for a particular model
        
    Returns:
        dict: Consistency evaluation results
    """
    valid_responses = []
    json_responses = []
    
    for result in model_results:
        if 'error' not in result and 'response' in result:
            valid_responses.append(result['response'])
            json_obj = extract_json_from_text(result['response'])
            if json_obj:
                json_responses.append(json_obj)
    
    if not valid_responses:
        return {
            "format_consistency": 0,
            "json_validity": 0,
            "valid_json_percentage": 0,
            "json_structure_consistency": 0
        }
    
    # Calculate format consistency
    format_similarity_sum = 0
    comparisons = 0
    for i in range(len(valid_responses)):
        for j in range(i+1, len(valid_responses)):
            similarity = SequenceMatcher(None, valid_responses[i], valid_responses[j]).ratio()
            format_similarity_sum += similarity
            comparisons += 1
    
    format_consistency = format_similarity_sum / comparisons if comparisons > 0 else 0
    
    # Calculate JSON validity
    json_validity = sum(1 for r in valid_responses if is_valid_json(r)) / len(valid_responses) if valid_responses else 0
    
    # Calculate JSON structure consistency
    json_structure_consistency = 0
    if len(json_responses) >= 2:
        structure_sim_sum = 0
        structure_comparisons = 0
        for i in range(len(json_responses)):
            for j in range(i+1, len(json_responses)):
                # Use shared keys to measure structure similarity
                shared_keys = set(json_responses[i].keys()) & set(json_responses[j].keys())
                total_keys = set(json_responses[i].keys()) | set(json_responses[j].keys())
                if total_keys:
                    structure_sim_sum += len(shared_keys) / len(total_keys)
                    structure_comparisons += 1
        
        json_structure_consistency = structure_sim_sum / structure_comparisons if structure_comparisons > 0 else 0
    
    return {
        "format_consistency": format_consistency,
        "json_validity": json_validity,
        "valid_json_percentage": (len(json_responses) / len(valid_responses)) * 100 if valid_responses else 0,
        "json_structure_consistency": json_structure_consistency
    }

def evaluate_model(model_results, expected_structure=None):
    """
    Evaluate the quality of model responses
    
    Args:
        model_results: List of results for a particular model
        expected_structure: Optional expected JSON structure for comparison
        
    Returns:
        dict: Model evaluation metrics
    """
    logger.info(f"\n============================================================")
    logger.info(f"📋 EVALUATING MODEL RESULTS")
    logger.info(f"============================================================")
    logger.info(f"Number of results to evaluate: {len(model_results) if model_results else 0}")
    
    if not model_results:
        logger.warning(f"⚠️ No model results to evaluate")
        return {
            "success_rate": 0,
            "avg_latency": 0,
            "avg_cost": 0,
            "cost_per_1000": 0,
            "json_validity": 0,
            "format_consistency": 0,
            "json_structure_consistency": 0,
            "structural_accuracy": 0
        }
    
    # Log all results for debugging
    for i, result in enumerate(model_results):
        logger.info(f"📄 Result {i+1} details:")
        if 'error' in result:
            logger.error(f"  - ERROR: {result.get('error', 'Unknown error')}")
        else:
            logger.info(f"  - Has response: {bool(result.get('response'))}")
            logger.info(f"  - Response length: {len(result.get('response', '')) if result.get('response') else 0} chars")
            logger.info(f"  - Tokens: {result.get('total_tokens', 0)} total")
            logger.info(f"  - Latency: {result.get('latency', 0):.2f}s")
            logger.info(f"  - Cost: ${result.get('cost', 0):.6f}")
    
    # Filter out errors
    successful_results = [r for r in model_results if 'error' not in r and 'response' in r]
    logger.info(f"Number of successful results: {len(successful_results)}/{len(model_results)}")
    
    if not successful_results:
        logger.warning(f"⚠️ No successful results found (all had errors or missing responses)")
        return {
            "success_rate": 0,
            "avg_latency": 0,
            "avg_cost": 0,
            "cost_per_1000": 0,
            "json_validity": 0,
            "format_consistency": 0,
            "json_structure_consistency": 0,
            "structural_accuracy": 0
        }
    
    # Extract basic statistics
    success_rate = len(successful_results) / len(model_results)
    logger.info(f"Success rate: {success_rate:.2f} ({len(successful_results)}/{len(model_results)})")
    
    avg_latency = sum(r['latency'] for r in successful_results) / len(successful_results)
    logger.info(f"Average latency: {avg_latency:.2f} seconds")
    
    # Calculate token usage metrics
    total_prompt_tokens = sum(r.get('prompt_tokens', 0) for r in successful_results)
    total_completion_tokens = sum(r.get('completion_tokens', 0) for r in successful_results)
    total_tokens = sum(r.get('total_tokens', 0) for r in successful_results)
    
    avg_prompt_tokens = total_prompt_tokens / len(successful_results)
    avg_completion_tokens = total_completion_tokens / len(successful_results)
    avg_tokens = total_tokens / len(successful_results)
    
    logger.info(f"Token usage metrics:")
    logger.info(f"  - Average input tokens: {avg_prompt_tokens:.1f}")
    logger.info(f"  - Average output tokens: {avg_completion_tokens:.1f}")
    logger.info(f"  - Average total tokens: {avg_tokens:.1f}")
    logger.info(f"  - Total input tokens: {total_prompt_tokens}")
    logger.info(f"  - Total output tokens: {total_completion_tokens}")
    
    avg_cost = sum(r['cost'] for r in successful_results) / len(successful_results)
    logger.info(f"Average cost: ${avg_cost:.6f} per request")
    
    # Evaluate consistency
    logger.info(f"📋 Evaluating response consistency...")
    consistency_metrics = evaluate_consistency(successful_results)
    logger.info(f"Consistency metrics:")
    logger.info(f"  - JSON validity: {consistency_metrics['json_validity']:.2f}")
    logger.info(f"  - Format consistency: {consistency_metrics['format_consistency']:.2f}")
    logger.info(f"  - JSON structure consistency: {consistency_metrics['json_structure_consistency']:.2f}")
    
    # Evaluate structural accuracy if expected structure is provided
    structural_accuracy = 0
    if expected_structure:
        logger.info(f"📋 Evaluating structural accuracy against expected structure...")
        structure_scores = []
        for i, result in enumerate(successful_results):
            logger.info(f"Checking structure for result {i+1}:")
            json_obj = extract_json_from_text(result['response'])
            if json_obj:
                logger.info(f"  - Successfully extracted JSON")
                score = compare_json_structure(json_obj, expected_structure)
                logger.info(f"  - Structure score: {score:.2f}")
                structure_scores.append(score)
            else:
                logger.warning(f"  - Failed to extract JSON from response")
        
        structural_accuracy = sum(structure_scores) / len(structure_scores) if structure_scores else 0
        logger.info(f"Overall structural accuracy: {structural_accuracy:.2f}")
    
    cost_per_1000 = avg_cost * 1000
    
    # Prepare the metrics dictionary to return
    metrics = {
        "success_rate": success_rate,
        "avg_latency": avg_latency,
        "avg_cost": avg_cost,
        "cost_per_1000": cost_per_1000,
        "avg_tokens": avg_tokens,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "json_validity": consistency_metrics['json_validity'],
        "format_consistency": consistency_metrics['format_consistency'],
        "json_structure_consistency": consistency_metrics['json_structure_consistency'],
        "structural_accuracy": structural_accuracy
    }
    
    logger.info(f"✅ EVALUATION COMPLETE")
    logger.info(f"============================================================")
    
    return metrics

def evaluate_results(results, expected_structure=None):
    """
    Evaluate results from multiple models
    
    Args:
        results: Dictionary mapping model names to their results
        expected_structure: Optional expected JSON structure for comparison
        
    Returns:
        dict: Evaluation results for each model
    """
    logger.info(f"\n============================================================")
    logger.info(f"📊 BEGINNING MULTI-MODEL EVALUATION")
    logger.info(f"============================================================")
    logger.info(f"Number of models to evaluate: {sum(1 for k in results if not k.endswith('_stats'))}")
    logger.info(f"Models: {[k for k in results if not k.endswith('_stats')]}")
    
    evaluation = {}
    
    for model_name, model_results in results.items():
        # Skip the stats entries
        if model_name.endswith('_stats'):
            logger.info(f"Skipping stats entry: {model_name}")
            continue
        
        logger.info(f"🔍 Evaluating model: {model_name}")
        evaluation[model_name] = evaluate_model(model_results, expected_structure)
    
    # Rank models based on different metrics
    logger.info(f"🏆 Ranking models based on metrics...")
    ranking = rank_models(evaluation)
    evaluation["ranking"] = ranking
    
    logger.info(f"\n============================================================")
    logger.info(f"✅ MULTI-MODEL EVALUATION COMPLETE")
    logger.info(f"============================================================")
    
    return evaluation

def rank_models(evaluation):
    """
    Rank models based on different metrics
    
    Args:
        evaluation: Dictionary of model evaluations
        
    Returns:
        dict: Rankings for each metric
    """
    metrics = [
        "success_rate",
        "avg_latency",
        "avg_cost",
        "json_validity",
        "format_consistency",
        "json_structure_consistency",
        "structural_accuracy"
    ]
    
    rankings = {}
    
    for metric in metrics:
        if metric in ["avg_latency", "avg_cost"]:
            # Lower is better for these metrics
            sorted_models = sorted(
                evaluation.items(),
                key=lambda x: x[1].get(metric, float('inf'))
            )
        else:
            # Higher is better for these metrics
            sorted_models = sorted(
                evaluation.items(),
                key=lambda x: x[1].get(metric, 0),
                reverse=True
            )
        
        rankings[metric] = [model for model, _ in sorted_models]
    
    # Calculate overall rank
    overall_scores = {}
    for model in evaluation:
        overall_scores[model] = 0
        for metric in metrics:
            # For each metric, award points based on rank
            if model in rankings[metric]:
                rank = rankings[metric].index(model)
                if metric in ["avg_latency", "avg_cost"]:
                    # Lower is better, so give more points for lower ranks
                    overall_scores[model] += len(rankings[metric]) - rank
                else:
                    # Higher is better, so give more points for higher ranks
                    overall_scores[model] += len(rankings[metric]) - rank
    
    # Sort models by overall score
    overall_ranking = sorted(
        overall_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    rankings["overall"] = [model for model, _ in overall_ranking]
    
    return rankings
