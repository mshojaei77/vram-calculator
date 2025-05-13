# Setup Guide for VRAM Calculator

This guide provides detailed instructions for setting up and using the VRAM Calculator for Large Language Models (LLMs).

## System Requirements

- Python 3.8 or higher
- Windows, macOS, or Linux operating system
- Internet connection for initial setup

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/vram_calculator.git
cd vram_calculator
```

### Step 2: Create a Virtual Environment (Recommended)

#### Using venv (built into Python)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### Using conda

```bash
conda create -n vram_calc python=3.10
conda activate vram_calc
```

### Step 3: Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Using UV (faster installation)
pip install uv
uv pip install -r requirements.txt
```

### Step 4: Configure Environment Variables (Optional)

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` to customize the default settings.

## Usage

### Running the Streamlit Web Application

The Streamlit interface provides a user-friendly way to use the VRAM calculator:

```bash
streamlit run app.py
```

Your browser should automatically open to the application. If not, navigate to the URL displayed in the terminal (usually http://localhost:8501).

### Using the Command Line Interface

For quick calculations or scripting, use the CLI:

```bash
# Interactive mode
python vram_calculator.py

# Direct command with arguments
python vram_calculator.py --model-size 7 --precision INT8 --include-kv-cache --context-length 4096 --use-case inference
```

## Common Issues and Troubleshooting

### StreamlitAPIException

If you encounter an error related to Streamlit:

```
Check that you have streamlit installed: pip install streamlit
```

### ImportError: No module named 'dotenv'

Install the python-dotenv package:

```bash
pip install python-dotenv
```

### Display Issues in the Web Interface

If charts or UI elements don't display properly, try:

```bash
streamlit cache clear
```

## Tips for Accurate Calculations

1. For actual model architecture specifications, check the model card or documentation
2. The KV cache can be disabled for basic calculations focused only on model weights
3. When fine-tuning, account for larger batch sizes that may increase VRAM usage
4. For large models (70B+), multi-GPU setups are typically necessary

## Updating the Calculator

To update to the latest version:

```bash
git pull
pip install -r requirements.txt
```

## Getting Help

If you encounter issues or need assistance:
- Check the [documentation](doc.md)
- Submit an issue on GitHub
- Contribute a pull request to improve the tool 