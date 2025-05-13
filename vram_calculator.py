import os
import argparse
from dotenv import load_dotenv
import math

# Load environment variables
load_dotenv()

def calculate_model_size_vram(params_billions, precision):
    """Calculate VRAM required for model weights based on precision"""
    precision_bytes = {
        "FP32": 4.0,
        "FP16": 2.0,
        "INT8": 1.0,
        "INT4": 0.5,
        "INT3": 0.375,
        "INT2": 0.25
    }
    
    # Convert billions to actual parameter count
    params = params_billions * 1_000_000_000
    
    # Calculate base VRAM in bytes
    vram_bytes = params * precision_bytes[precision]
    
    # Convert to GB and add 20% overhead
    vram_gb = (vram_bytes / (1024**3)) * 1.2
    
    return vram_gb

def calculate_kv_cache(num_layers, hidden_dim, seq_length, batch_size=1, precision="FP16"):
    """Calculate KV cache size based on model architecture and context length"""
    bytes_per_value = 2 if precision == "FP16" else 4  # Assuming FP16 or FP32
    
    # KV Cache formula: 2 × Layers × Hidden_Dim × Seq_Length × Batch_Size × Bytes_per_Value
    kv_cache_bytes = 2 * num_layers * hidden_dim * seq_length * batch_size * bytes_per_value
    
    # Convert to GB
    kv_cache_gb = kv_cache_bytes / (1024**3)
    
    return kv_cache_gb

def calculate_total_vram(model_vram, kv_cache_vram, use_case="inference", include_kv_cache=False):
    """Calculate total VRAM based on model size, KV cache, and use case"""
    
    # Add basic overhead
    total_vram = model_vram
    
    # Add KV cache if enabled
    if include_kv_cache:
        total_vram += kv_cache_vram
    
    # Additional overhead based on use case
    if use_case == "full_fine_tuning":
        # Full fine-tuning requires much more memory (optimizer states, gradients)
        total_vram = total_vram * 2.5
    elif use_case == "lora":
        # LoRA requires less memory than full fine-tuning
        total_vram = total_vram * 1.5
    elif use_case == "qlora":
        # QLoRA is more efficient than LoRA
        total_vram = total_vram * 1.2
    
    # Add framework overhead (0.5-1GB)
    total_vram += 0.75
    
    return total_vram

def recommend_gpu(required_vram):
    """Provide GPU recommendations based on calculated VRAM requirements"""
    
    if required_vram <= 4:
        return "Entry Level (4GB): NVIDIA GTX 1650 (4GB) / RTX 3050 Mobile (4GB) - Limited to small models (1-3B) with high quantization"
    elif required_vram <= 6:
        return "Budget (6GB): NVIDIA RTX 2060 (6GB) / GTX 1660 Ti (6GB) - Suitable for 3-7B models with 4-bit quantization"
    elif required_vram <= 8:
        return "Mid-Budget (8GB): NVIDIA RTX 3050 (8GB) / RTX 4060 (8GB) - Good for 7B models with proper quantization"
    elif required_vram <= 10:
        return "Upper-Mid (10GB): NVIDIA RTX 3080 (10GB) / RTX 2080 Ti (11GB) - Can handle 7-13B models with quantization"
    elif required_vram <= 16:
        return "Mid-Range: NVIDIA RTX A4000 (16GB) / RTX 2000 Ada (16GB) / Tesla V100 (16GB)"
    elif required_vram <= 20:
        return "Mid-High Range: NVIDIA RTX 4000 Ada (20GB) / RTX A4500 (20GB)"
    elif required_vram <= 24:
        return "High-End Consumer: NVIDIA RTX 4090 (24GB) / RTX 3090 (24GB) / RTX 3090 Ti (24GB) / RTX A5000 (24GB) / L4 (24GB) / A30 (24GB)"
    elif required_vram <= 32:
        return "Enthusiast: NVIDIA RTX 5090 (32GB)"
    elif required_vram <= 48:
        return "Professional: NVIDIA RTX A6000 (48GB) / RTX 6000 Ada (48GB) / A40 (48GB) / L40 (48GB) / L40S (48GB)"
    elif required_vram <= 80:
        return "Datacenter: NVIDIA A100 (80GB) / H100 PCIe (80GB) / H100 SXM (80GB)"
    elif required_vram <= 94:
        return "High-End Datacenter: NVIDIA H100 NVL (94GB)"
    elif required_vram <= 141:
        return "Advanced Datacenter: NVIDIA H200 SXM (141GB)"
    elif required_vram <= 180:
        return "Ultimate AI: NVIDIA B200 (180GB)"
    else:
        # Multi-GPU recommendations based on required VRAM
        multi_gpu_options = []
        
        # Option 1: Highest performance with newest GPUs
        if required_vram <= 360:  # 2 x B200
            multi_gpu_options.append("2 × NVIDIA B200 (180GB each)")
        elif required_vram <= 540:  # 3 x B200
            multi_gpu_options.append("3 × NVIDIA B200 (180GB each)")
        elif required_vram <= 720:  # 4 x B200
            multi_gpu_options.append("4 × NVIDIA B200 (180GB each)")
        else:
            multi_gpu_options.append(f"{math.ceil(required_vram / 180)} × NVIDIA B200 (180GB each)")
            
        # Option 2: High performance with H200
        if required_vram <= 282:  # 2 x H200
            multi_gpu_options.append("2 × NVIDIA H200 SXM (141GB each)")
        elif required_vram <= 423:  # 3 x H200
            multi_gpu_options.append("3 × NVIDIA H200 SXM (141GB each)")
        elif required_vram <= 564:  # 4 x H200
            multi_gpu_options.append("4 × NVIDIA H200 SXM (141GB each)")
        else:
            multi_gpu_options.append(f"{math.ceil(required_vram / 141)} × NVIDIA H200 SXM (141GB each)")
            
        # Option 3: More accessible with H100
        if required_vram <= 160:  # 2 x H100
            multi_gpu_options.append("2 × NVIDIA H100 PCIe/SXM (80GB each)")
        elif required_vram <= 240:  # 3 x H100
            multi_gpu_options.append("3 × NVIDIA H100 PCIe/SXM (80GB each)")
        elif required_vram <= 320:  # 4 x H100
            multi_gpu_options.append("4 × NVIDIA H100 PCIe/SXM (80GB each)")
        elif required_vram <= 640:  # 8 x H100
            multi_gpu_options.append(f"{math.ceil(required_vram / 80)} × NVIDIA H100 PCIe/SXM (80GB each)")
        else:
            multi_gpu_options.append(f"{math.ceil(required_vram / 80)} × NVIDIA H100 PCIe/SXM (80GB each)")
            
        # Option 4: More economical with A100
        if required_vram <= 160:  # 2 x A100
            multi_gpu_options.append("2 × NVIDIA A100 (80GB each)")
        elif required_vram <= 320:  # 4 x A100
            multi_gpu_options.append("4 × NVIDIA A100 (80GB each)")
        elif required_vram <= 640:  # 8 x A100
            multi_gpu_options.append("8 × NVIDIA A100 (80GB each)")
        else:
            multi_gpu_options.append(f"{math.ceil(required_vram / 80)} × NVIDIA A100 (80GB each)")
        
        # Option 5: Consumer/workstation option with RTX A6000/etc.
        if required_vram <= 96:  # 2 x RTX A6000/A40/etc.
            multi_gpu_options.append("2 × NVIDIA RTX A6000/A40/L40 (48GB each)")
        elif required_vram <= 144:  # 3 x RTX A6000/A40/etc.
            multi_gpu_options.append("3 × NVIDIA RTX A6000/A40/L40 (48GB each)")
        elif required_vram <= 192:  # 4 x RTX A6000/A40/etc.
            multi_gpu_options.append("4 × NVIDIA RTX A6000/A40/L40 (48GB each)")
        elif required_vram <= 384:  # 8 x RTX A6000/A40/etc.
            multi_gpu_options.append(f"{math.ceil(required_vram / 48)} × NVIDIA RTX A6000/A40/L40 (48GB each)")
        
        # Option 6: Consumer option with RTX 4090/3090
        if required_vram <= 48:  # 2 x RTX 4090/3090
            multi_gpu_options.append("2 × NVIDIA RTX 4090/3090 (24GB each)")
        elif required_vram <= 96:  # 4 x RTX 4090/3090
            multi_gpu_options.append("4 × NVIDIA RTX 4090/3090 (24GB each)")
        elif required_vram <= 192:  # 8 x RTX 4090/3090
            multi_gpu_options.append("8 × NVIDIA RTX 4090/3090 (24GB each)")
        
        recommendation = f"Multi-GPU setup required. Total VRAM needed: {required_vram:.2f} GB\n\nRecommended options:\n"
        recommendation += "\n".join([f"• {option}" for option in multi_gpu_options])
        
        return recommendation

def estimate_model_architecture(model_size):
    """Estimate model architecture based on model size"""
    estimated_layers = int(2.5 * model_size) if model_size >= 7 else int(3 * model_size)
    estimated_hidden_dim = 128 * round(model_size * 64) if model_size >= 7 else 128 * round(model_size * 96)
    
    return estimated_layers, estimated_hidden_dim

def get_user_input():
    """Get model and usage parameters from user"""
    print("\n=== VRAM Calculator for Large Language Models ===\n")
    
    # Model parameters
    model_size = float(input("Enter model size in billions of parameters (e.g., 7 for 7B): "))
    
    # Precision option
    print("\nSelect precision/quantization:")
    print("1. FP32 (32-bit)")
    print("2. FP16 (16-bit)")
    print("3. INT8 (8-bit)")
    print("4. INT4 (4-bit)")
    print("5. INT3 (3-bit)")
    print("6. INT2 (2-bit)")
    precision_option = int(input("Enter option (1-6): "))
    
    precision_map = {1: "FP32", 2: "FP16", 3: "INT8", 4: "INT4", 5: "INT3", 6: "INT2"}
    precision = precision_map.get(precision_option, "FP16")
    
    # KV cache inclusion option
    include_kv_cache_input = input("\nInclude KV cache in VRAM calculations? (y/n, default: n): ").lower()
    include_kv_cache = include_kv_cache_input == 'y'
    
    # Context length (only if KV cache is included)
    context_length = 2048  # default
    if include_kv_cache:
        context_length = int(input("\nEnter context length (number of tokens, e.g., 2048): "))
    else:
        print("\nSkipping context length as KV cache is disabled")
    
    # Use case
    print("\nSelect use case:")
    print("1. Inference only")
    print("2. Full fine-tuning")
    print("3. LoRA fine-tuning")
    print("4. QLoRA fine-tuning")
    use_case_option = int(input("Enter option (1-4): "))
    
    use_case_map = {1: "inference", 2: "full_fine_tuning", 3: "lora", 4: "qlora"}
    use_case = use_case_map.get(use_case_option, "inference")
    
    # For KV cache calculation - get model details
    # For simplicity, we'll estimate layers and hidden dimension based on model size
    estimated_layers, estimated_hidden_dim = estimate_model_architecture(model_size)
    
    # Option to override estimated architecture
    print(f"\nEstimated model architecture: {estimated_layers} layers, {estimated_hidden_dim} hidden dimension")
    override = input("Do you want to override these values? (y/n): ").lower()
    
    if override == 'y':
        num_layers = int(input("Enter number of layers: "))
        hidden_dim = int(input("Enter hidden dimension size: "))
    else:
        num_layers = estimated_layers
        hidden_dim = estimated_hidden_dim
    
    return {
        "model_size": model_size,
        "precision": precision,
        "context_length": context_length,
        "use_case": use_case,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "include_kv_cache": include_kv_cache
    }

def calculate_and_display_results(params):
    """Calculate and display VRAM requirements based on parameters"""
    # Calculate model size VRAM
    model_vram = calculate_model_size_vram(params["model_size"], params["precision"])
    
    # Calculate KV cache VRAM
    kv_cache_vram = calculate_kv_cache(
        params["num_layers"], 
        params["hidden_dim"], 
        params["context_length"],
        precision=params["precision"]
    )
    
    # Calculate total VRAM
    include_kv_cache = params.get("include_kv_cache", False)
    total_vram = calculate_total_vram(model_vram, kv_cache_vram, params["use_case"], include_kv_cache)
    
    # Print results
    print("\n=== VRAM Calculation Results ===")
    print(f"Model parameters: {params['model_size']:.2f}B")
    print(f"Precision: {params['precision']}")
    print(f"Context length: {params['context_length']} tokens")
    print(f"Use case: {params['use_case']}")
    print(f"Model architecture: {params['num_layers']} layers, {params['hidden_dim']} hidden dimension")
    print(f"Include KV cache in calculation: {include_kv_cache}")
    
    print("\nVRAM Requirements:")
    print(f"Model weights: {model_vram:.2f} GB")
    if include_kv_cache:
        print(f"KV cache: {kv_cache_vram:.2f} GB")
    else:
        print(f"KV cache: Disabled (not included in total)")
    print(f"Total VRAM needed: {total_vram:.2f} GB")
    
    # Provide GPU recommendation
    recommendation = recommend_gpu(total_vram)
    print(f"\nRecommended GPU: {recommendation}")
    
    # Additional information for context
    if params["use_case"] != "inference":
        print("\nNote: Training/fine-tuning requires significantly more VRAM than inference.")
        print("Consider using parameter-efficient fine-tuning methods if VRAM is limited.")
    
    if include_kv_cache and params["context_length"] > 8192:
        print("\nTip: For long contexts, using Flash Attention can reduce VRAM usage.")
    
    return {
        "model_vram": model_vram,
        "kv_cache_vram": kv_cache_vram,
        "total_vram": total_vram,
        "recommendation": recommendation
    }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Calculate VRAM requirements for LLMs")
    parser.add_argument("--model-size", type=float, help="Model size in billions of parameters")
    parser.add_argument("--precision", choices=["FP32", "FP16", "INT8", "INT4", "INT3", "INT2"],
                       default="FP16", help="Model precision/quantization")
    parser.add_argument("--context-length", type=int, help="Context length in tokens")
    parser.add_argument("--use-case", choices=["inference", "full_fine_tuning", "lora", "qlora"],
                       default="inference", help="Use case (inference or fine-tuning method)")
    parser.add_argument("--layers", type=int, help="Number of model layers")
    parser.add_argument("--hidden-dim", type=int, help="Hidden dimension size")
    parser.add_argument("--include-kv-cache", action="store_true", help="Include KV cache in VRAM calculations (default: disabled)")
    
    return parser.parse_args()

def main():
    # Check if arguments are provided
    args = parse_args()
    
    if args.model_size:
        # Use command line arguments
        model_size = args.model_size
        precision = args.precision
        context_length = args.context_length or 2048
        use_case = args.use_case
        include_kv_cache = args.include_kv_cache
        
        # Estimate or use provided architecture details
        if args.layers and args.hidden_dim:
            num_layers = args.layers
            hidden_dim = args.hidden_dim
        else:
            num_layers, hidden_dim = estimate_model_architecture(model_size)
        
        params = {
            "model_size": model_size,
            "precision": precision,
            "context_length": context_length,
            "use_case": use_case,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "include_kv_cache": include_kv_cache
        }
    else:
        # Get parameters from interactive CLI
        params = get_user_input()
    
    # Calculate and display results
    calculate_and_display_results(params)

if __name__ == "__main__":
    main() 