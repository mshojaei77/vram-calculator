# GPU VRAM Calculator for Large Language Models (LLMs)

This tool helps you calculate VRAM requirements for running Large Language Models (LLMs) and recommends suitable GPUs based on your needs. It provides both a command-line interface and a Streamlit web application for ease of use.

## Features

- Calculate VRAM requirements for LLMs based on:
  - Model size (parameters)
  - Precision/quantization level
  - Context length (optional)
  - Use case (inference or fine-tuning)
- Option to include/exclude KV cache in calculations (default: excluded)
- Visualize VRAM usage breakdown with interactive charts
- Compare VRAM usage across different precision levels
- Analyze impact of context length on memory requirements
- Get GPU recommendations based on calculated VRAM needs
- Multi-GPU recommendations for large models
- Detailed GPU comparison guide for all VRAM capacities (4GB to 180GB)

## Documentation

For comprehensive information about GPU selection for LLMs, please refer to our [detailed guide](doc.md) that covers:
- Understanding GPU requirements for LLMs
- Key GPU specifications to consider
- Memory usage fundamentals
- VRAM requirements by model size
- GPU recommendations by budget
- Optimization techniques
- And much more!

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/vram_calculator.git
   cd vram_calculator
   ```

2. Install dependencies using UV (preferred) or pip:
   ```bash
   # Using UV
   uv pip install -r requirements.txt
   
   # Using pip
   pip install -r requirements.txt
   ```

## Usage

### Streamlit Web Application

Run the Streamlit app for an interactive GUI experience:

```bash
streamlit run app.py
```

This will launch the web application in your default browser, providing an intuitive interface for:
- Adjusting all model parameters with sliders and input fields
- Toggling KV cache inclusion in calculations
- Viewing real-time calculations
- Visualizing VRAM usage with charts
- Getting detailed GPU recommendations
- Exploring multi-GPU options for larger models

### Command Line Interface

For quick calculations, you can use the command-line interface:

#### Interactive Mode

```bash
python vram_calculator.py
```

Follow the prompts to input your model parameters.

#### Arguments Mode

```bash
python vram_calculator.py --model-size 7 --precision INT8 --include-kv-cache --context-length 4096 --use-case inference
```

Available arguments:
- `--model-size`: Model size in billions of parameters (e.g., 7 for a 7B model)
- `--precision`: Precision/quantization (FP32, FP16, INT8, INT4, INT3, INT2)
- `--include-kv-cache`: Include KV cache in VRAM calculations (default: disabled)
- `--context-length`: Context length in tokens (only used if KV cache is included)
- `--use-case`: Use case (inference, full_fine_tuning, lora, qlora)
- `--layers`: Optional - Number of model layers (estimated if not provided)
- `--hidden-dim`: Optional - Hidden dimension size (estimated if not provided)

## Understanding VRAM Requirements

### Key Factors Affecting VRAM Usage

1. **Model Size**: More parameters require more VRAM
2. **Precision/Quantization**: Lower-precision models use less VRAM
3. **Context Length**: Larger context windows increase VRAM usage via KV cache
4. **Training vs. Inference**: Training requires significantly more VRAM

### Precision and Memory Usage

| Format | Bytes per Parameter | Memory Reduction (vs FP16) |
|--------|---------------------|----------------------------|
| FP32 (32-bit) | 4.0 | -100% (2× more memory) |
| FP16/BF16 (16-bit) | 2.0 | Baseline |
| INT8 (8-bit) | 1.0 | 50% |
| INT4 (4-bit) | 0.5 | 75% |
| INT3 (3-bit) | 0.375 | 81.25% |
| INT2 (2-bit) | 0.25 | 87.5% |

## Formulas Used

### Model Weights VRAM
```
VRAM (bytes) = Number of Parameters × Bytes per Parameter × Overhead
```

### KV Cache VRAM (when enabled)
```
KV Cache (bytes) = 2 × Layers × Hidden_Dim × Seq_Length × Batch_Size × Bytes_per_Value
```

### Fine-tuning Overhead
- Full fine-tuning: 2.5× model weights + KV cache (if enabled)
- LoRA fine-tuning: 1.5× model weights + KV cache (if enabled)
- QLoRA fine-tuning: 1.2× model weights + KV cache (if enabled)

## Multi-GPU Configurations

For models requiring more VRAM than available on a single GPU, the calculator provides:
- Multiple options based on different GPU types
- Exact number of GPUs needed based on VRAM requirements
- Recommendations for different budget levels and use cases
- Information about parallelization strategies

## Configuration

The application uses environment variables for configuration, which can be set in a `.env` file:

```
# Example .env file
DEBUG=False
CUSTOM_RECOMMENDATIONS=False
DEFAULT_PRECISION=FP16
DEFAULT_USE_CASE=inference
INCLUDE_KV_CACHE=False
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Made with ❤️ for the LLM community
