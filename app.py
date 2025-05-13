import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import markdown
from dotenv import load_dotenv
from vram_calculator import (
    calculate_model_size_vram,
    calculate_kv_cache,
    calculate_total_vram,
    recommend_gpu
)

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="LLM VRAM Calculator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to read the documentation file
def load_documentation():
    try:
        with open("doc.md", "r") as file:
            return file.read()
    except Exception as e:
        return f"Error loading documentation: {str(e)}"

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2196F3;
        margin-top: 20px;
    }
    .result-box {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .documentation-link {
        margin-top: 15px;
        margin-bottom: 25px;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<p class="main-header">GPU VRAM Calculator for Large Language Models</p>', unsafe_allow_html=True)

st.markdown("""
This tool helps you calculate the VRAM requirements for running Large Language Models (LLMs) and recommends suitable GPUs based on your needs.
""")

# Documentation link
st.markdown('<div class="documentation-link">📚 For comprehensive information about GPU selection, check the <a href="#" onclick="document.getElementById(\'documentation-tab\').click()">detailed documentation</a>.</div>', unsafe_allow_html=True)

# Create tabs for calculator and documentation
calculator_tab, documentation_tab = st.tabs(["VRAM Calculator", "GPU Selection Guide"])

with calculator_tab:
    # Create sidebar for inputs
    with st.sidebar:
        st.markdown("## Model Parameters")
        
        # Model size
        model_size = st.number_input(
            "Model Size (billions of parameters)",
            min_value=0.1,
            max_value=1000.0,
            value=7.0,
            step=0.1,
            format="%.1f",
            help="Number of parameters in billions (e.g., 7 for a 7B model)"
        )
        
        # Precision
        precision_options = {
            "FP32 (32-bit)": "FP32",
            "FP16 (16-bit)": "FP16", 
            "INT8 (8-bit)": "INT8",
            "INT4 (4-bit)": "INT4",
            "INT3 (3-bit)": "INT3",
            "INT2 (2-bit)": "INT2"
        }
        precision = st.selectbox(
            "Precision/Quantization",
            options=list(precision_options.keys()),
            index=1,  # Default to FP16
            help="Lower precision reduces VRAM usage but may affect model quality"
        )
        precision_value = precision_options[precision]
        
        # Include KV Cache option
        include_kv_cache = st.checkbox(
            "Include KV Cache in calculation",
            value=False,
            help="Enable to include context window memory usage in VRAM calculations"
        )
        
        # Context length - only enabled if KV cache is included
        context_length = st.number_input(
            "Context Length (tokens)",
            min_value=128,
            max_value=1000000,
            value=2048,
            step=128,
            help="Maximum number of tokens in the context window",
            disabled=not include_kv_cache
        )
        
        # Use case
        use_case_options = {
            "Inference only": "inference",
            "Full fine-tuning": "full_fine_tuning",
            "LoRA fine-tuning": "lora",
            "QLoRA fine-tuning": "qlora"
        }
        use_case = st.selectbox(
            "Use Case",
            options=list(use_case_options.keys()),
            index=0,  # Default to inference
            help="Training/fine-tuning requires more VRAM than inference"
        )
        use_case_value = use_case_options[use_case]
        
        # Advanced options
        with st.expander("Advanced Model Architecture", expanded=False):
            # Estimate layers and hidden dimension based on model size
            estimated_layers = int(2.5 * model_size) if model_size >= 7 else int(3 * model_size)
            estimated_hidden_dim = 128 * round(model_size * 64) if model_size >= 7 else 128 * round(model_size * 96)
            
            st.info(f"Estimated architecture: {estimated_layers} layers, {estimated_hidden_dim} hidden dimension")
            
            override_architecture = st.checkbox("Override estimated architecture", value=False)
            
            if override_architecture:
                num_layers = st.number_input("Number of Layers", min_value=1, max_value=200, value=estimated_layers)
                hidden_dim = st.number_input(
                    "Hidden Dimension", 
                    min_value=128, 
                    max_value=8192, 
                    value=estimated_hidden_dim,
                    step=128
                )
            else:
                num_layers = estimated_layers
                hidden_dim = estimated_hidden_dim

    # Calculate button in main content area
    if st.button("Calculate VRAM Requirements", type="primary", use_container_width=True):
        # Calculate model size VRAM
        model_vram = calculate_model_size_vram(model_size, precision_value)
        
        # Calculate KV cache VRAM
        kv_cache_vram = calculate_kv_cache(
            num_layers, 
            hidden_dim, 
            context_length,
            precision=precision_value
        )
        
        # Calculate total VRAM
        total_vram = calculate_total_vram(model_vram, kv_cache_vram, use_case_value, include_kv_cache)
        
        # GPU recommendation
        recommendation = recommend_gpu(total_vram)
        
        # Display results
        st.markdown('<p class="section-header">VRAM Calculation Results</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.subheader("Input Parameters")
            st.write(f"**Model size:** {model_size:.1f}B parameters")
            st.write(f"**Precision:** {precision_value}")
            if include_kv_cache:
                st.write(f"**Context length:** {context_length:,} tokens")
            else:
                st.write("**Context length:** Disabled (KV cache not included)")
            st.write(f"**Use case:** {use_case}")
            st.write(f"**Model architecture:** {num_layers} layers, {hidden_dim} hidden dimension")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.subheader("VRAM Requirements")
            st.write(f"**Model weights:** {model_vram:.2f} GB")
            if include_kv_cache:
                st.write(f"**KV cache:** {kv_cache_vram:.2f} GB")
            else:
                st.write("**KV cache:** Disabled (not included in total)")
            st.write(f"**Total VRAM needed:** {total_vram:.2f} GB")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # GPU recommendation
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.subheader("GPU Recommendation")
        if "\n" in recommendation:
            # Handle multi-line recommendations (for multi-GPU setups)
            st.markdown(recommendation)
        else:
            # Handle single-line recommendations (for single GPU)
            st.markdown(f"### {recommendation}")
        
        # Additional information based on context
        if use_case_value != "inference":
            st.info("**Note:** Training/fine-tuning requires significantly more VRAM than inference. Consider using parameter-efficient fine-tuning methods if VRAM is limited.")
        
        if include_kv_cache and context_length > 8192:
            st.info("**Tip:** For long contexts, using Flash Attention can reduce VRAM usage.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create visualization
        st.markdown('<p class="section-header">VRAM Usage Breakdown</p>', unsafe_allow_html=True)
        
        # Prepare data for pie chart
        if include_kv_cache:
            labels = ['Model Weights', 'KV Cache', 'Framework Overhead']
            sizes = [model_vram, kv_cache_vram, total_vram - (model_vram + kv_cache_vram)]
        else:
            labels = ['Model Weights', 'Framework Overhead']
            sizes = [model_vram, total_vram - model_vram]
        
        # Create figure with matplotlib
        if include_kv_cache:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart
        if include_kv_cache:
            ax1.pie(sizes, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#2196F3', '#FFC107'])
        else:
            ax1.pie(sizes, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FFC107'])
        ax1.axis('equal')
        ax1.set_title('VRAM Usage Distribution')
        ax1.legend(labels, loc="best")
        
        # Bar chart for different precision options
        precisions = ["FP32", "FP16", "INT8", "INT4", "INT3", "INT2"]
        vram_values = []
        
        for p in precisions:
            model_v = calculate_model_size_vram(model_size, p)
            if include_kv_cache:
                kv_v = calculate_kv_cache(num_layers, hidden_dim, context_length, precision=p)
            else:
                kv_v = 0
            total_v = calculate_total_vram(model_v, kv_v, use_case_value, include_kv_cache)
            vram_values.append(total_v)
        
        bars = ax2.bar(precisions, vram_values, color='#2196F3')
        ax2.set_title('VRAM Usage by Precision')
        ax2.set_ylabel('VRAM (GB)')
        ax2.set_xlabel('Precision')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom')
        
        st.pyplot(fig)
        
        # Comparison table for different context lengths - only show if KV cache is included
        if include_kv_cache:
            st.markdown('<p class="section-header">Context Length Impact</p>', unsafe_allow_html=True)
            
            context_sizes = [2048, 4096, 8192, 16384, 32768, 65536]
            context_data = []
            
            for ctx in context_sizes:
                kv_v = calculate_kv_cache(num_layers, hidden_dim, ctx, precision=precision_value)
                total_v = calculate_total_vram(model_vram, kv_v, use_case_value, include_kv_cache=True)
                context_data.append({
                    "Context Length": f"{ctx:,} tokens",
                    "KV Cache Size": f"{kv_v:.2f} GB",
                    "Total VRAM": f"{total_v:.2f} GB"
                })
            
            st.table(pd.DataFrame(context_data))

    # Add information section
    with st.expander("📚 Learn More About VRAM Requirements", expanded=False):
        st.markdown("""
        ### Key Factors Affecting VRAM Usage
        
        1. **Model Size**: More parameters require more VRAM
        2. **Precision/Quantization**: Lower-precision models use less VRAM
        3. **Context Length**: Larger context windows increase VRAM usage via KV cache
        4. **Training vs. Inference**: Training requires significantly more VRAM
        
        ### Precision and Quantization
        
        | Format | Bytes per Parameter | Memory Reduction (vs FP16) |
        |--------|---------------------|----------------------------|
        | FP32 (32-bit) | 4.0 | -100% (2× more memory) |
        | FP16/BF16 (16-bit) | 2.0 | Baseline |
        | INT8 (8-bit) | 1.0 | 50% |
        | INT4 (4-bit) | 0.5 | 75% |
        | INT3 (3-bit) | 0.375 | 81.25% |
        | INT2 (2-bit) | 0.25 | 87.5% |
        
        ### Optimization Techniques
        
        - **Flash Attention**: Reduces memory complexity for attention mechanisms
        - **PagedAttention**: Optimizes KV cache storage and management
        - **Parameter-Efficient Fine-Tuning**: Methods like LoRA and QLoRA reduce training VRAM requirements
        - **Quantization Algorithms**: GPTQ, AWQ, and HQQ offer better quality-to-size ratios
        """)
    
    # Add GPU comparison section
    with st.expander("🖥️ GPU Comparison Guide", expanded=False):
        st.markdown("""
        ### Available NVIDIA GPUs by VRAM Capacity
        
        #### Datacenter / AI GPUs
        
        | GPU Model | VRAM | Typical Use Case |
        |-----------|------|------------------|
        | NVIDIA B200 | 180GB | Largest LLMs (100B+), multi-task AI systems |
        | NVIDIA H200 SXM | 141GB | Very large LLMs (70B+), multi-modal AI |
        | NVIDIA H100 NVL | 94GB | Large-scale LLMs, training with long contexts |
        | NVIDIA H100 PCIe/SXM | 80GB | Training large LLMs, high-throughput inference |
        | NVIDIA A100 PCIe/SXM | 80GB | Training and inference of larger models |
        
        #### Professional / Workstation GPUs
        
        | GPU Model | VRAM | Typical Use Case |
        |-----------|------|------------------|
        | NVIDIA A40/L40/L40S | 48GB | Professional workloads, medium-scale training |
        | NVIDIA RTX A6000 | 48GB | Professional visualization, medium-scale AI |
        | NVIDIA RTX 6000 Ada | 48GB | Professional workloads, deep learning |
        | NVIDIA RTX 5090 | 32GB | High-end enthusiast, 30B model inference |
        | NVIDIA A30 | 24GB | Enterprise inference, smaller training tasks |
        | NVIDIA L4 | 24GB | Efficient inference, small-scale training |
        | NVIDIA RTX A5000 | 24GB | Professional visualization, ML development |
        
        #### Consumer / Gaming GPUs
        
        | GPU Model | VRAM | Typical Use Case |
        |-----------|------|------------------|
        | NVIDIA RTX 4090 | 24GB | High-end gaming, ML research, inference |
        | NVIDIA RTX 3090/3090 Ti | 24GB | Gaming, affordable ML development |
        | NVIDIA RTX A4500 | 20GB | Mid-range professional visualization |
        | NVIDIA RTX 4000 Ada | 20GB | Balanced professional workloads |
        | NVIDIA RTX A4000 | 16GB | Entry professional visualization |
        | NVIDIA Tesla V100 | 16GB | Legacy datacenter AI |
        | NVIDIA RTX 2000 Ada | 16GB | Entry professional workloads |
        | NVIDIA RTX 2080 Ti | 11GB | Gaming, entry-level ML tasks |
        | NVIDIA RTX 3080 | 10GB | Gaming, basic ML tasks |
        | NVIDIA RTX 3060 | 12GB | Budget gaming, basic AI development |
        | NVIDIA RTX 4060/4060 Ti | 8GB | Modern gaming, entry-level AI |
        | NVIDIA RTX 3050 | 8GB | Budget gaming, basic AI experiments |
        | NVIDIA RTX 2060/GTX 1660 Ti | 6GB | Legacy gaming, very basic ML tasks |
        | NVIDIA GTX 1650 | 4GB | Entry-level gaming, extremely limited ML |
        
        ### LLM Size Capability by GPU
        
        | Model Size | Recommended GPUs for Inference (INT8/4-bit) |
        |------------|-------------------------------------------|
        | 0.5-1B     | GTX 1650+ (4GB+) |
        | 1-3B       | RTX 2060+ (6GB+) |
        | 3-7B       | RTX 3050/4060+ (8GB+) |
        | 7-13B      | RTX 4090, RTX A5000, L4 (24GB) |
        | 13-30B     | RTX 5090 (32GB), A40, RTX A6000 (48GB) |
        | 30-70B     | A100, H100 (80GB) |
        | 70B+       | H100 NVL (94GB), H200 (141GB), B200 (180GB) |
        
        ### Small LLM Use Cases (< 10GB VRAM)
        
        | VRAM | Model Size | Quantization | Typical Applications |
        |------|------------|--------------|----------------------|
        | 4GB  | 0.5-1B     | INT4/INT3    | Small chatbots, text classification, embeddings |
        | 6GB  | 1-3B       | INT4         | Simple assistants, code completion, basic RAG |
        | 8GB  | 3-7B       | INT4/INT8    | General-purpose assistants, creative writing, summarization |
        | 10GB | 7-10B      | INT8/INT4    | Advanced reasoning, instruction following, complex tasks |
        
        *Note: For fine-tuning, significantly more VRAM is required compared to inference.*
        """)
    
    # Add Multi-GPU section
    with st.expander("🔄 Multi-GPU Configurations & Parallelization", expanded=False):
        st.markdown("""
        ### Multi-GPU Parallelization Strategies
        
        When a model is too large to fit on a single GPU, several parallelization strategies can be employed:
        
        #### 1. Tensor Parallelism
        
        **Best for:** Models that exceed single GPU memory but need fast inference
        
        **How it works:** 
        - Splits individual layers across multiple GPUs
        - Each GPU holds a portion of each layer's weights
        - Requires high-speed GPU interconnects (NVLink/NVSwitch)
        
        **Ideal GPU combinations:**
        - Matching GPUs (same model) for balanced computation
        - GPUs with NVLink support (e.g., A100, H100, RTX 3090)
        
        #### 2. Pipeline Parallelism
        
        **Best for:** Very deep models
        
        **How it works:**
        - Different layers are assigned to different GPUs
        - Data flows through GPUs sequentially 
        - Reduces communication overhead compared to tensor parallelism
        
        **Ideal GPU combinations:**
        - Can work with mixed GPU types, but matching is preferred
        - Works well with GPUs that have fast interconnects
        
        #### 3. Expert Parallelism (for MoE models)
        
        **Best for:** Mixture-of-Experts models
        
        **How it works:**
        - Different experts are placed on different GPUs
        - Only active experts are computed for each input
        
        **Ideal GPU combinations:**
        - Works well with heterogeneous GPU setups
        - Large VRAM GPUs for dense computation parts
        
        #### 4. Data Parallelism
        
        **Best for:** Training, or batch inference
        
        **How it works:**
        - Full model copy on each GPU
        - Each GPU processes different data
        
        **Ideal GPU combinations:**
        - Can use mixed GPU types but may be limited by slowest GPU
        - Best with matching GPUs for balanced batches
        
        ### Best GPU Combinations for Multi-GPU Setups
        
        #### For Maximum Performance
        - **4× or 8× NVIDIA H100 SXM5** with NVLink/NVSwitch
        - **4× or 8× NVIDIA A100 SXM4** with NVLink/NVSwitch
        - **2× NVIDIA B200** for the largest models
        
        #### For Workstation/Research
        - **4× NVIDIA RTX A6000** (with NVLink bridges)
        - **4× NVIDIA RTX 4090** (limited by PCIe bandwidth)
        - **2× NVIDIA A40/L40** (with NVLink when available)
        
        #### Most Cost-Effective
        - **4× NVIDIA RTX 3090** (limited by PCIe bandwidth)
        - **2× NVIDIA RTX A6000** (with NVLink)
        - **8× NVIDIA RTX 4090** (for distributed training)
        
        ### Software Frameworks for Multi-GPU Inference
        
        - **DeepSpeed**: Offers ZeRO-Inference for efficient multi-GPU inference
        - **vLLM**: Supports tensor parallelism for faster inference
        - **Hugging Face Accelerate**: Simple API for distributed inference
        - **PyTorch FSDP**: Fully Sharded Data Parallel for training and inference
        - **Megatron-LM**: Advanced tensor and pipeline parallelism strategies
        
        ### Key Considerations
        
        1. **Bandwidth matters:** NVLink-connected GPUs achieve much higher performance for tensor parallelism
        2. **Matching GPUs:** Using identical GPUs ensures balanced computation
        3. **Power and cooling:** Multi-GPU setups require robust power supplies and cooling solutions
        4. **Software support:** Not all frameworks support all parallelization strategies
        """)

with documentation_tab:
    st.markdown('<p class="section-header">GPU Selection Guide for Large Language Models</p>', unsafe_allow_html=True)
    st.markdown(load_documentation())

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit • [View the code on GitHub](https://github.com/) • [View full documentation](doc.md)") 