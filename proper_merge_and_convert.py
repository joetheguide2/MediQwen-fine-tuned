"""
Properly merge LoRA adapters and convert to GGUF for Ollama
This ensures the GGUF file contains your fine-tuned weights
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import sys

def merge_lora_to_full_model():
    """Merge LoRA adapters into base model - EXACTLY like inference_terminal.py"""
    
    print("="*80)
    print("STEP 1: Loading Tokenizer and Base Model")
    print("="*80)
    
    # Load tokenizer FIRST (same as inference_terminal.py)
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "./qwen1.5b-clinical-reasoning",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"✅ Tokenizer vocabulary size: {len(tokenizer)}")
    
    # Load base model in full precision (FP16) on CPU
    print("\n📦 Loading base model (this may take a few minutes)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float16,  # FP16 for GGUF conversion
        device_map="cpu",  # CPU to avoid GPU memory issues
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Resize embeddings to match tokenizer (CRITICAL!)
    print(f"\n🔧 Resizing model embeddings to {len(tokenizer)} tokens...")
    base_model.resize_token_embeddings(len(tokenizer))
    
    print("="*80)
    print("STEP 2: Loading and Merging LoRA Adapters")
    print("="*80)
    
    # Load LoRA adapters
    print("\n🔧 Loading LoRA adapters from ./qwen1.5b-clinical-reasoning/adapters/")
    adapter_path = "./qwen1.5b-clinical-reasoning/adapters"
    
    if not os.path.exists(adapter_path):
        print(f"❌ ERROR: Adapter path not found: {adapter_path}")
        sys.exit(1)
    
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("✅ LoRA adapters loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading adapters: {e}")
        sys.exit(1)
    
    # Merge LoRA weights into base model
    print("\n🔀 Merging LoRA weights into base model...")
    print("   (This creates a single model with your fine-tuned weights)")
    model = model.merge_and_unload()
    print("✅ LoRA weights merged successfully!")
    
    print("="*80)
    print("STEP 3: Saving Merged Model")
    print("="*80)
    
    # Save merged model
    output_dir = "./qwen-clinical-merged-fp16"
    print(f"\n💾 Saving merged model to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    
    print("✅ Merged model saved!")
    
    # Verify the save
    import glob
    safetensors_files = glob.glob(f"{output_dir}/*.safetensors")
    print(f"\n📊 Saved files:")
    for f in safetensors_files:
        size_mb = os.path.getsize(f) / (1024**2)
        print(f"   {os.path.basename(f)}: {size_mb:.1f} MB")
    
    return output_dir

def convert_to_gguf(merged_model_path):
    """Convert merged model to GGUF format"""
    
    print("\n" + "="*80)
    print("STEP 4: Converting to GGUF Format")
    print("="*80)
    
    # Check if llama.cpp exists
    llama_cpp_path = input("\nEnter path to llama.cpp directory [./llama.cpp]: ").strip() or "./llama.cpp"
    
    if not os.path.exists(llama_cpp_path):
        print(f"\n❌ llama.cpp not found at: {llama_cpp_path}")
        print("\nPlease install llama.cpp first:")
        print("  git clone https://github.com/ggerganov/llama.cpp")
        print("  cd llama.cpp && make")
        print(f"\nYour merged model is ready at: {merged_model_path}")
        print("You can convert it later using llama.cpp")
        return None
    
    convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
    quantize_bin = os.path.join(llama_cpp_path, "llama-quantize")
    
    if not os.path.exists(convert_script):
        print(f"❌ Conversion script not found: {convert_script}")
        return None
    
    print("\n📋 Available quantization levels:")
    print("   Q4_K_M  - 4-bit (recommended for mobile, ~400MB)")
    print("   Q5_K_M  - 5-bit (better quality, ~500MB)")
    print("   Q8_0    - 8-bit (high quality, ~800MB)")
    print("   F16     - 16-bit (best quality, ~3GB)")
    
    quant_type = input("\nEnter quantization [Q4_K_M]: ").strip() or "Q4_K_M"
    
    output_gguf = f"qwen-clinical-{quant_type.lower()}.gguf"
    
    print(f"\n🔄 Converting to GGUF...")
    
    import subprocess
    
    try:
        # Convert to F16 GGUF first
        temp_f16 = "qwen-clinical-f16-temp.gguf"
        print(f"⏳ Step 1/2: Converting to F16 GGUF...")
        
        result = subprocess.run([
            sys.executable,
            convert_script,
            merged_model_path,
            "--outfile", temp_f16,
            "--outtype", "f16"
        ], check=True, capture_output=True, text=True)
        
        print("✅ F16 GGUF created")
        
        # Quantize if not F16
        if quant_type.upper() != "F16":
            print(f"⏳ Step 2/2: Quantizing to {quant_type}...")
            
            if not os.path.exists(quantize_bin):
                print(f"❌ Quantization binary not found: {quantize_bin}")
                print(f"Using F16 version: {temp_f16}")
                output_gguf = temp_f16
            else:
                result = subprocess.run([
                    quantize_bin,
                    temp_f16,
                    output_gguf,
                    quant_type
                ], check=True, capture_output=True, text=True)
                
                print(f"✅ Quantized to {quant_type}")
                
                # Remove temp file
                if os.path.exists(temp_f16):
                    os.remove(temp_f16)
        else:
            output_gguf = temp_f16
        
        size_mb = os.path.getsize(output_gguf) / (1024**2)
        print(f"\n✅ GGUF file created: {output_gguf} ({size_mb:.1f} MB)")
        
        return output_gguf
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None

def create_modelfile(gguf_path):
    """Create Ollama Modelfile that matches inference_terminal.py"""
    
    print("\n" + "="*80)
    print("STEP 5: Creating Ollama Modelfile")
    print("="*80)
    
    # Get absolute path
    abs_gguf_path = os.path.abspath(gguf_path)
    
    modelfile_content = f"""# Clinical Reasoning Model - Properly Merged with LoRA
FROM {abs_gguf_path}

# EXACT template from inference_terminal.py
TEMPLATE \"\"\"<|im_start|>user
Analyze this clinical note and provide diagnostic reasoning:

Clinical Note:
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"

# Parameters matching inference_terminal.py
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 50
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 200
PARAMETER num_ctx 2048

# Stop tokens
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

# NO system message - not in training
SYSTEM ""
"""
    
    modelfile_path = "Modelfile.qwen-clinical-merged"
    
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"✅ Modelfile created: {modelfile_path}")
    print(f"\n📄 Contents:")
    print("-" * 80)
    print(modelfile_content)
    print("-" * 80)
    
    return modelfile_path

def main():
    print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║     PROPER LORA MERGE AND GGUF CONVERSION FOR OLLAMA                           ║
║     This ensures your fine-tuned weights are in the GGUF file                  ║
╚════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Step 1-3: Merge LoRA
        merged_path = merge_lora_to_full_model()
        
        # Step 4: Convert to GGUF
        gguf_path = convert_to_gguf(merged_path)
        
        if gguf_path:
            # Step 5: Create Modelfile
            modelfile_path = create_modelfile(gguf_path)
            
            print("\n" + "="*80)
            print("✅ CONVERSION COMPLETE!")
            print("="*80)
            
            print(f"""
📦 Your files:
   - Merged model: {merged_path}/
   - GGUF file: {gguf_path}
   - Modelfile: {modelfile_path}

🚀 Next steps:
   1. Create Ollama model:
      ollama create qwen-clinical -f {modelfile_path}
   
   2. Test it:
      ollama run qwen-clinical "Chest pain

Patient is 65yo male with HTN, diabetes. Troponin-T 0.55"
   
   3. It should now match inference_terminal.py results! ✅
            """)
        else:
            print("\n⚠️  GGUF conversion skipped or failed")
            print(f"Merged model available at: {merged_path}")
            print("You can convert it manually using llama.cpp")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()