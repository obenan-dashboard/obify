import os
import json
import time
from datetime import datetime
# Fix matplotlib threading issues by using non-GUI backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent threading issues
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO

def format_cost(cost_value, precision=6):
    """
    Format a cost value with appropriate precision based on its magnitude
    
    Args:
        cost_value: The cost value to format
        precision: Default precision for regular values
        
    Returns:
        str: Formatted cost string with $ prefix
    """
    # Ensure we have a numeric value
    try:
        cost_value = float(cost_value)
    except (TypeError, ValueError):
        cost_value = 0.0
        
    # Handle very small positive values with higher precision
    if cost_value < 0.0001 and cost_value > 0:
        return f"${cost_value:.6f}"
    # Handle small values with moderate precision
    elif cost_value < 0.01 and cost_value > 0:
        return f"${cost_value:.4f}"
    # Handle zero values specially - show a minimal value instead of $0.00
    elif cost_value == 0:
        return "$0.0001"
    # Handle regular values with standard precision
    else:
        return f"${cost_value:.{precision}f}"

def create_bar_chart(data, title, labels, values_key, filename=None):
    """
    Create a bar chart from the given data
    
    Args:
        data: Dictionary mapping labels to values
        title: Title of the chart
        labels: List of labels for the x-axis
        values_key: Key in the data dictionary to use for values
        filename: Optional filename to save the chart to
        
    Returns:
        str: Base64-encoded image data if no filename is provided
    """
    plt.figure(figsize=(10, 6))
    
    # Get values from the data dictionary
    values = [data[label].get(values_key, 0) if label in data else 0 for label in labels]
    
    # Special handling for cost values - they can be very small numbers
    if values_key == 'cost_per_1000' or values_key == 'avg_cost':
        # Ensure we have non-zero values by setting a minimum threshold
        # This prevents bars from disappearing in the chart
        min_visible_value = 0.00001  # Minimum threshold to ensure visibility
        values = [max(v, min_visible_value) for v in values]
        
    x = np.arange(len(labels))
    
    bars = plt.bar(x, values, width=0.6)
    
    # Add value labels above each bar for better visibility
    for i, bar in enumerate(bars):
        value = values[i]
        if values_key == 'cost_per_1000' or values_key == 'avg_cost':
            # Format cost values with appropriate precision
            if value < 0.01:
                label_text = f'${value:.6f}'
            else:
                label_text = f'${value:.2f}'
        else:
            label_text = f'{value:.2f}'
            
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 label_text, ha='center', va='bottom', fontsize=8, rotation=45)
    
    # Handle model names with or without provider prefixes
    model_labels = []
    for label in labels:
        if ':' in label:
            model_labels.append(label.split(':')[1])  # Get part after provider prefix
        else:
            model_labels.append(label)  # Use full name if no prefix
    
    plt.xticks(x, model_labels, rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        plt.close()
        return filename
    else:
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode()
        plt.close()
        return img_data

def create_comparison_chart(data, title, labels, metrics, filename=None):
    """
    Create a grouped bar chart comparing multiple metrics across models
    
    Args:
        data: Dictionary mapping labels to values
        title: Title of the chart
        labels: List of labels for the x-axis
        metrics: List of metrics to compare
        filename: Optional filename to save the chart to
        
    Returns:
        str: Base64-encoded image data if no filename is provided
    """
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(labels))
    width = 0.2
    offset = -width * (len(metrics) / 2)
    
    for i, metric in enumerate(metrics):
        values = [data[label].get(metric, 0) if label in data else 0 for label in labels]
        plt.bar(x + offset + i*width, values, width, label=metric)
    
    plt.xlabel('Models')
    
    # Handle model names with or without provider prefixes
    model_labels = []
    for label in labels:
        if ':' in label:
            model_labels.append(label.split(':')[1])  # Get part after provider prefix
        else:
            model_labels.append(label)  # Use full name if no prefix
    
    plt.xticks(x, model_labels, rotation=45, ha='right')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        plt.close()
        return filename
    else:
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode()
        plt.close()
        return img_data

def generate_charts(evaluation, results, report_folder, report_name):
    """
    Generate charts for the report and save them to files
    
    Args:
        evaluation: Dictionary of model evaluations
        results: Dictionary of model results
        report_folder: Folder to save the charts to
        report_name: Base name for the chart files
        
    Returns:
        dict: Mapping of chart types to their filenames
    """
    os.makedirs(os.path.join(report_folder, 'charts'), exist_ok=True)
    
    charts = {}
    models = [model for model in evaluation if model != 'ranking']
    
    # Cost comparison chart
    cost_chart_file = os.path.join(report_folder, 'charts', f"{report_name}_cost.png")
    charts['cost'] = create_bar_chart(
        evaluation,
        'Cost per 1000 Reviews ($)',
        models,
        'cost_per_1000',
        cost_chart_file
    )
    
    # Latency comparison chart
    latency_chart_file = os.path.join(report_folder, 'charts', f"{report_name}_latency.png")
    charts['latency'] = create_bar_chart(
        evaluation,
        'Average Response Time (seconds)',
        models,
        'avg_latency',
        latency_chart_file
    )
    
    # Success rate chart
    success_chart_file = os.path.join(report_folder, 'charts', f"{report_name}_success.png")
    charts['success'] = create_bar_chart(
        evaluation,
        'Success Rate',
        models,
        'success_rate',
        success_chart_file
    )
    
    # Quality metrics comparison
    quality_metrics = ['json_validity', 'format_consistency', 'json_structure_consistency']
    quality_chart_file = os.path.join(report_folder, 'charts', f"{report_name}_quality.png")
    charts['quality'] = create_comparison_chart(
        evaluation,
        'Quality Metrics Comparison',
        models,
        quality_metrics,
        quality_chart_file
    )
    
    return charts

def format_model_summary(model, stats, evaluation):
    """
    Format a summary of model performance
    
    Args:
        model: Model name
        stats: Statistics for the model
        evaluation: Evaluation results for the model
        
    Returns:
        dict: Formatted model summary
    """
    # Handle model names with or without a provider prefix
    if ':' in model:
        _, model_name = model.split(':')
    else:
        model_name = model
    
    return {
        'model_name': model_name,
        'full_name': model,
        'success_rate': f"{evaluation.get('success_rate', 0) * 100:.2f}%",
        'avg_latency': f"{evaluation.get('avg_latency', 0):.2f} seconds",
        'avg_cost': format_cost(evaluation.get('avg_cost', 0)),
        'cost_per_1000': format_cost(evaluation.get('cost_per_1000', 0), precision=2),
        'quality_metrics': {
            'json_validity': f"{evaluation.get('json_validity', 0) * 100:.2f}%",
            'format_consistency': f"{evaluation.get('format_consistency', 0) * 100:.2f}%",
            'json_structure_consistency': f"{evaluation.get('json_structure_consistency', 0) * 100:.2f}%"
        },
        'total_reviews': stats.get('total_reviews', 0),
        'successful_reviews': stats.get('successful', 0),
        'error_count': stats.get('errors', 0),
        'avg_tokens': int(stats.get('avg_tokens', 0)),
        'avg_input_tokens': int(stats.get('avg_prompt_tokens', 0)),
        'avg_output_tokens': int(stats.get('avg_completion_tokens', 0)),
        'total_input_tokens': int(stats.get('total_prompt_tokens', 0)),
        'total_output_tokens': int(stats.get('total_completion_tokens', 0))
    }

def generate_report(results, evaluation, original_filename, report_name, report_folder):
    """
    Generate a comprehensive report of the model evaluations
    
    Args:
        results: Dictionary of model results
        evaluation: Dictionary of model evaluations
        original_filename: Name of the original file containing reviews
        report_name: Name for the report
        report_folder: Folder to save the report to
        
    Returns:
        str: Path to the saved report
    """
    os.makedirs(report_folder, exist_ok=True)
    
    # Generate charts
    charts = generate_charts(evaluation, results, report_folder, report_name)
    
    # Prepare report data
    report = {
        'report_name': report_name,
        'timestamp': datetime.now().isoformat(),
        'source_file': original_filename,
        'charts': charts,
        'rankings': evaluation.get('ranking', {}),
        'models': {}
    }
    
    # Add model summaries
    for model in [m for m in results if not m.endswith('_stats')]:
        stats = results.get(f"{model}_stats", {})
        model_eval = evaluation.get(model, {})
        report['models'][model] = format_model_summary(model, stats, model_eval)
    
    # Calculate overall recommendations
    if 'ranking' in evaluation and 'overall' in evaluation['ranking'] and evaluation['ranking']['overall']:
        top_models = evaluation['ranking']['overall'][:3] if len(evaluation['ranking']['overall']) >= 3 else evaluation['ranking']['overall']
        report['recommendations'] = {
            'best_overall': top_models[0] if top_models else None,
            'top_models': top_models,
            'best_value': evaluation['ranking'].get('avg_cost', [])[0] if evaluation['ranking'].get('avg_cost') and evaluation['ranking']['avg_cost'] else None,
            'fastest': evaluation['ranking'].get('avg_latency', [])[0] if evaluation['ranking'].get('avg_latency') and evaluation['ranking']['avg_latency'] else None,
            'most_accurate': evaluation['ranking'].get('json_validity', [])[0] if evaluation['ranking'].get('json_validity') and evaluation['ranking']['json_validity'] else None
        }
    else:
        # Fallback if no rankings are available
        report['recommendations'] = {
            'best_overall': None,
            'top_models': [],
            'best_value': None,
            'fastest': None,
            'most_accurate': None
        }
    
    # Generate all_responses with full details for each model response
    all_responses = {}
    for model in [m for m in results if not m.endswith('_stats')]:
        model_results = results.get(model, [])
        if model_results:
            # Store all results from this model
            model_responses = []
            
            for idx, result in enumerate(model_results):
                # Format the response details
                cost = result.get('cost', 0)
                if cost < 0.000001 and cost > 0:
                    cost_str = f"${cost:.8f}"
                else:
                    cost_str = f"${cost:.6f}"
                
                # Get the full review and prompt that were used
                review_text = result.get('full_review', result.get('review', 'No review text available'))
                
                # Get the full prompt that was sent to the model
                prompt_sent = result.get('full_prompt', result.get('prompt', 'Prompt not recorded'))
                
                response_details = {
                    'review_number': idx + 1,
                    'review_text': review_text,
                    'prompt_sent': prompt_sent,
                    'response': result.get('response', 'No response available'),
                    'success': result.get('success', False),
                    'tokens': result.get('total_tokens', 0),
                    'cost': cost_str,
                    'latency': f"{result.get('latency', 0):.2f} seconds"
                }
                
                model_responses.append(response_details)
            
            all_responses[model] = model_responses
    
    # Add all responses to the report
    report['all_responses'] = all_responses
    
    # Add sample responses for backward compatibility
    sample_responses = {}
    for model, responses in all_responses.items():
        if responses:
            # Use first successful response as sample
            successful_responses = [r for r in responses if r.get('success', False)]
            sample = successful_responses[0] if successful_responses else responses[0]
            sample_responses[model] = {
                'review': sample.get('review_text', 'No review text available'),
                'response': sample.get('response', 'No response available'),
                'tokens': sample.get('tokens', 0),
                'cost': sample.get('cost', '$0.000000'),
                'latency': sample.get('latency', '0.00 seconds')
            }
    
    report['sample_responses'] = sample_responses
    
    # Save the report
    report_path = os.path.join(report_folder, f"{report_name}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report_path
