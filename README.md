# Obify

<div align="center">

![Obify Logo](static/images/logo.png)

**Enterprise-grade AI model comparison platform for review analysis**

</div>

## 📋 Overview

Obify is a powerful platform that enables enterprises to systematically compare the performance, cost, and quality of multiple AI language models for review processing and text analysis. With Obify, you can make data-driven decisions about which AI models best fit your specific use cases.

### 🎬 Demo

Check out our [demo video](demo.MP4) to see Obify in action.

## ✨ Features

- **Multi-Model Evaluation**: Test multiple language models side-by-side with standardized prompts
- **Performance Metrics**: Compare success rates, latency, cost efficiency, and response quality 
- **Dynamic Review Extraction**: Upload files with intelligent column detection that automatically identifies review content
- **Enterprise Reporting**: Generate detailed reports with interactive visualizations and exportable data
- **Cost Analytics**: Track token usage and calculate exact costs for each model based on current pricing

## 🤖 Supported Models

### OpenAI
- GPT-4
- GPT-4o
- GPT-4o-mini
- GPT-3.5-turbo
- o1
- o1-mini
- o3
- o3-mini

### Anthropic Claude
- claude-3-opus
- claude-3-sonnet
- claude-3-haiku
- claude-3-5-sonnet
- claude-3-7-sonnet

### Google Gemini
- Gemini-2.5-pro
- Gemini-2.5-flash
- Gemini-2.5-flash-lite
- Gemini-2.0-flash
- Gemini-2.0-flash-lite

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- API keys for the models you want to test (OpenAI, Anthropic Claude, and/or Google Gemini)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/obenan-dashboard/obify.git
   cd obify
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**:
   
   Create a `config.json` file in the root directory with the following structure:
   ```json
   {
     "api_keys": {
       "openai": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
       "anthropic": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
       "google": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
     },
     "note": ""
   }
   ```
   
   > **Note:** Replace the placeholders with your actual API keys. The config.json file is included in .gitignore to prevent accidental commits of your API keys.

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Access the web interface**:
   
   Open your browser and navigate to `http://127.0.0.1:5005`

## 📊 Usage

1. **Upload Data**: On the home page, upload your CSV, TXT, or Excel file containing reviews
2. **Select Review Column**: If multiple columns are detected, you'll be prompted to select which one contains review text
3. **Configure Test**: Choose which models to compare, set your prompt template, and start the analysis
4. **View Results**: Explore comparative performance metrics, sample responses, and interactive charts
5. **Export Report**: Generate PDF or download complete results for reporting and sharing

## 🔧 Extending Obify

Obify is designed to be easily extended with new models and providers. You can:

- Add support for new models from existing providers
- Update pricing information
- Implement new pricing structures
- Add entirely new providers

### How to Add a New Model

To add a new model, edit the `utils/model_config.py` file. This centralized config file contains all model definitions and pricing information. For example:

```python
# Add a new model to an existing provider
OPENAI_MODELS["new-model-name"] = {
    "name": "New Model Display Name",
    "input_price": 0.000XX,  # per token price
    "output_price": 0.000XX, # per token price
    "icon": "bi-robot"       # Bootstrap icon class
}
```

See the [Contributing Guide](docs/CONTRIBUTING_MODELS.md) for detailed instructions on extending model support.

## 📁 Project Structure

- `app.py` - Main Flask application
- `utils/` - Core modules for model handling, file processing, and reporting
  - `model_handler.py` - API integrations and model invocations
  - `model_config.py` - Centralized model definitions and pricing
  - `file_handler.py` - File upload and review extraction logic
  - `report_generator.py` - Metrics calculation and report generation
  - `evaluator.py` - Quality metrics and response evaluation
- `static/` - CSS, JavaScript, and images
- `templates/` - HTML templates for the web interface
- `docs/` - Documentation files

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or support, please open an issue on the GitHub repository.

---

<div align="center">

**Obify** - Compare AI models with confidence

</div>

## 🔧 Extending Obify

Obify is designed to be easily extended with new models and providers. You can:

- Add support for new models from existing providers
- Update pricing information
- Implement new pricing structures
- Add entirely new providers

### How to Add a New Model

To add a new model, edit the `utils/model_config.py` file. This centralized config file contains all model definitions and pricing information. For example:

```python
# Add a new model to an existing provider
OPENAI_MODELS["new-model-name"] = {
    "name": "New Model Display Name",
    "input_price": 0.000XX,  # per token price
    "output_price": 0.000XX, # per token price
    "icon": "bi-robot"       # Bootstrap icon class
}
```

See the [Contributing Guide](docs/CONTRIBUTING_MODELS.md) for detailed instructions on extending model support.

## 📁 Project Structure

- `app.py` - Main Flask application
- `utils/` - Core modules for model handling, file processing, and reporting
  - `model_handler.py` - API integrations and model invocations
  - `model_config.py` - Centralized model definitions and pricing
  - `file_handler.py` - File upload and review extraction logic
  - `report_generator.py` - Metrics calculation and report generation
  - `evaluator.py` - Quality metrics and response evaluation
- `static/` - CSS, JavaScript, and images
- `templates/` - HTML templates for the web interface
- `docs/` - Documentation files

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or support, please open an issue on the GitHub repository.

---

<div align="center">

**Obify** - Compare AI models with confidence

</div>

## Pricing and Cost Calculation

Obify accurately calculates costs using the following formula:
```
Total Cost = (Input Tokens / 1,000,000 × Input Cost per 1M Tokens) + (Output Tokens / 1,000,000 × Output Cost per 1M Tokens)
```

The application includes up-to-date pricing for all supported models, ensuring that cost calculations are accurate. Special handling is included for tiered pricing models (like Gemini 2.5 Pro).

### Centralized Model Configuration

All model definitions, pricing information, and calculation formulas are centralized in a single file (`utils/model_config.py`), making it easy to:
- Add new models
- Update pricing for existing models
- Implement new pricing structures
- Add entirely new providers

See the [Contributing Guide](docs/CONTRIBUTING_MODELS.md) for details on how to extend the model support.

## Installation

### Prerequisites

- Python 3.10+
- API keys for OpenAI and/or Anthropic Claude

### Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**:
   Create a `config.json` file in the root directory with the following structure:
   ```json
   {
       "api_keys": {
           "openai": "your-openai-api-key",
           "anthropic": "your-anthropic-api-key",
           "google": "your-gemini-api-key"
       },
       "note": ""
   }
   ```
   
   Alternatively, you can set environment variables:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   export GEMINI_API_KEY="your-gemini-api-key"
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Access the web interface**:
   Open your browser and navigate to `http://localhost:5000`

## Usage Guide

1. **Upload Dataset**: Start by uploading a CSV, TXT, or Excel file containing reviews or text to be processed
2. **Select Column**: If your file has multiple columns, select which column contains the reviews to analyze
3. **Select Models**: Choose one or more AI models to evaluate
4. **Configure Prompt**: Customize the prompt template that will be sent to the models
5. **Run Analysis**: Process your data through the selected models
6. **Review Results**: Examine the generated report with performance metrics and visualizations

## Contributing

Obify is designed to be easily extensible. To contribute:

1. **Adding New Models**: See the [Contributing Models Guide](docs/CONTRIBUTING_MODELS.md) for instructions on how to add new models or update existing ones
2. **Bug Fixes**: Submit pull requests with clear descriptions of the issue and solution
3. **Feature Requests**: Open an issue to discuss new features before implementing

All model definitions and pricing are centralized in `utils/model_config.py` to make contributions easy.

## Project Structure
```
Obify/
├── app.py              # Main Flask application
├── config.json        # API key configuration
├── requirements.txt   # Python dependencies
│
├── data/              # Data storage
│   ├── reports/       # Generated reports
│   └── uploads/       # Uploaded files
│
├── static/            # Static assets
│   ├── css/           # Stylesheets
│   ├── js/            # JavaScript files
│   └── images/        # Images and icons
│
├── templates/         # HTML templates
│   ├── index.html     # Main page
│   └── report.html    # Report template
│
└── utils/             # Utility modules
    ├── evaluator.py       # Model evaluation logic
    ├── file_handler.py    # File processing
    ├── model_handler.py   # API calls to language models
    └── report_generator.py  # Report generation
```

## Contributing

Contributions to Obify are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License.

## Credits

Developed by Jawad Ali at GSoft Consulting.

---

&copy; 2025 GSoft Consulting

MIT License
