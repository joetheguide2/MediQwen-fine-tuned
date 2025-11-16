"""
Convert Fine-tuned Model to Ollama-compatible GGUF Format
This script merges LoRA weights and converts to GGUF for use with Ollama on Android
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import subprocess
import sys
import json
from pathlib import Path

def print_step(step_num, message):
    """Print formatted step message"""
    print(f"\n{'='*80}")
    print(f"STEP {step_num}: {message}")
    print(f"{'='*80}\n")

def check_requirements():
    """Check and install required packages"""
    print_step(0, "Checking Requirements")
    
    required = {
        'transformers': 'transformers',
        'torch': 'torch',
        'peft': 'peft',
    }
    
    missing = []
    for package, pip_name in required.items():
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing.append(pip_name)
    
    if missing:
        print(f"\nInstalling missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("✅ All packages installed!")
    
    print("\n⚠️  NOTE: You need to install llama.cpp separately for GGUF conversion")
    print("Run: git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make")

def merge_lora_weights(
    base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path="./qwen1.5b-clinical-reasoning/adapters",
    output_path="./qwen-clinical-merged"
):
    """Merge LoRA weights into base model"""
    print_step(1, "Loading Base Model and Merging LoRA Weights")
    
    # CRITICAL: Load tokenizer first to get correct vocabulary size
    print("📝 Loading tokenizer to check vocabulary size...")
    tokenizer = AutoTokenizer.from_pretrained(
        "./qwen1.5b-clinical-reasoning",
        trust_remote_code=True
    )
    print(f"   Tokenizer vocabulary size: {len(tokenizer)}")
    
    print(f"\n📦 Loading base model: {base_model_name}")
    print("   (This will take a few minutes...)")
    
    # Load base model in full precision
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float16,  # Use float16 for compatibility
        device_map="cpu",  # Load on CPU to avoid GPU memory issues
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # CRITICAL FIX: Resize model to match tokenizer
    print(f"\n🔧 Resizing model embeddings to match tokenizer ({len(tokenizer)} tokens)...")
    base_model.resize_token_embeddings(len(tokenizer))
    
    print(f"\n🔧 Loading LoRA adapters from: {adapter_path}")
    
    # Load LoRA adapters
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("✅ LoRA adapters loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading adapters: {e}")
        print("Please check that the adapter path is correct.")
        sys.exit(1)
    
    print("\n🔀 Merging LoRA weights into base model...")
    print("   (This process may take 5-10 minutes...)")
    
    # Merge and unload LoRA weights
    model = model.merge_and_unload()
    
    print(f"\n💾 Saving merged model to: {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save merged model
    model.save_pretrained(output_path, safe_serialization=True)
    
    # Save tokenizer
    print("📝 Saving tokenizer...")
    tokenizer.save_pretrained(output_path)
    
    print(f"\n✅ Merged model saved successfully!")
    print(f"   Location: {os.path.abspath(output_path)}")
    
    return output_path

def convert_to_gguf(merged_model_path, output_name="qwen-clinical"):
    """Convert merged model to GGUF format"""
    print_step(2, "Converting to GGUF Format")
    
    print("📋 Available quantization levels:")
    print("   - Q4_K_M  : 4-bit quantization, medium quality (recommended for mobile)")
    print("   - Q5_K_M  : 5-bit quantization, higher quality")
    print("   - Q8_0    : 8-bit quantization, highest quality")
    print("   - F16     : 16-bit float (largest file, best quality)")
    
    quant_type = input("\nEnter quantization type [Q4_K_M]: ").strip() or "Q4_K_M"
    
    output_file = f"{output_name}-{quant_type.lower()}.gguf"
    
    # Check if llama.cpp is available
    llama_cpp_path = input("\nEnter path to llama.cpp directory [./llama.cpp]: ").strip() or "./llama.cpp"
    
    if not os.path.exists(llama_cpp_path):
        print(f"\n❌ llama.cpp not found at: {llama_cpp_path}")
        print("\n📥 Please install llama.cpp first:")
        print("   git clone https://github.com/ggerganov/llama.cpp")
        print("   cd llama.cpp")
        print("   make")
        print("\nOr download a pre-built release from: https://github.com/ggerganov/llama.cpp/releases")
        return None
    
    convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
    quantize_bin = os.path.join(llama_cpp_path, "llama-quantize")
    
    if not os.path.exists(convert_script):
        print(f"❌ Conversion script not found: {convert_script}")
        return None
    
    print(f"\n🔄 Converting to GGUF format...")
    print(f"   Input: {merged_model_path}")
    print(f"   Output: {output_file}")
    
    try:
        # First convert to GGUF F16
        temp_f16 = f"{output_name}-f16.gguf"
        print(f"\n⏳ Step 1/2: Converting to F16 GGUF...")
        subprocess.run([
            sys.executable,
            convert_script,
            merged_model_path,
            "--outfile", temp_f16,
            "--outtype", "f16"
        ], check=True)
        
        print(f"✅ F16 GGUF created: {temp_f16}")
        
        # Then quantize if not F16
        if quant_type.upper() != "F16":
            print(f"\n⏳ Step 2/2: Quantizing to {quant_type}...")
            if not os.path.exists(quantize_bin):
                print(f"❌ Quantization binary not found: {quantize_bin}")
                print("Please compile llama.cpp first (run 'make' in llama.cpp directory)")
                return temp_f16
            
            subprocess.run([
                quantize_bin,
                temp_f16,
                output_file,
                quant_type
            ], check=True)
            
            print(f"✅ Quantized GGUF created: {output_file}")
            
            # Remove temp F16 file
            if os.path.exists(temp_f16):
                os.remove(temp_f16)
                print(f"🗑️  Removed temporary file: {temp_f16}")
        else:
            output_file = temp_f16
        
        # Get file size
        file_size = os.path.getsize(output_file) / (1024 ** 2)  # Size in MB
        print(f"\n📊 Final model size: {file_size:.2f} MB")
        
        return output_file
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return None

def create_modelfile(gguf_path, output_name="qwen-clinical"):
    """Create Ollama Modelfile"""
    print_step(3, "Creating Ollama Modelfile")
    
    modelfile_content = f"""# Clinical Reasoning Model - Modelfile for Ollama
FROM {os.path.abspath(gguf_path)}

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 50
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

# System prompt for clinical reasoning
SYSTEM \"\"\"You are a clinical reasoning assistant trained on medical diagnostic data. 
Your task is to analyze clinical notes and provide structured diagnostic reasoning.
Format your response as: Diagnosis → Reasoning Step → Evidence.
Be concise and focus on the key diagnostic reasoning chain.\"\"\"

# Template for prompts
TEMPLATE \"\"\"<|im_start|>user
Analyze this clinical note and provide diagnostic reasoning:

Clinical Note:
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
\"\"\"
"""
    
    modelfile_path = f"Modelfile.{output_name}"
    
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"✅ Modelfile created: {modelfile_path}")
    print(f"\n📄 Modelfile contents:")
    print("-" * 80)
    print(modelfile_content)
    print("-" * 80)
    
    return modelfile_path

def create_ollama_instructions(gguf_path, modelfile_path, model_name="qwen-clinical"):
    """Create instructions for using with Ollama"""
    print_step(4, "Setup Instructions for Ollama")
    
    instructions = f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                    OLLAMA SETUP INSTRUCTIONS                                   ║
╚════════════════════════════════════════════════════════════════════════════════╝

📦 Your converted model files:
   • GGUF Model: {os.path.abspath(gguf_path)}
   • Modelfile:  {os.path.abspath(modelfile_path)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🖥️  DESKTOP SETUP (Linux/Mac/Windows):

1. Install Ollama:
   • Visit: https://ollama.ai
   • Download and install for your OS

2. Import the model:
   cd {os.path.dirname(os.path.abspath(modelfile_path))}
   ollama create {model_name} -f {os.path.basename(modelfile_path)}

3. Run the model:
   ollama run {model_name}

4. Test with a prompt:
   >>> Analyze this clinical note and provide diagnostic reasoning:
   
   Patient is a 65 year old male with chest pain...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📱 ANDROID SETUP:

1. Install Ollama Android App:
   • Download from: https://ollama.ai/download/android
   • Or build from source: https://github.com/ollama/ollama-android

2. Transfer model to Android:
   
   Option A - Using ADB:
   adb push {os.path.abspath(gguf_path)} /sdcard/Download/
   adb push {os.path.abspath(modelfile_path)} /sdcard/Download/
   
   Option B - Manual Transfer:
   • Copy both files to your phone via USB or cloud storage
   • Place them in: /sdcard/Ollama/models/ or accessible location

3. Import model in Ollama Android:
   • Open Ollama app
   • Tap "+" to add model
   • Select "Import from file"
   • Navigate to your GGUF file
   • Configure with the Modelfile settings

4. Alternative - Use Ollama API:
   • Set up Ollama on a server
   • Use the REST API from your Android app
   • Endpoint: http://your-server:11434/api/generate

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🧪 TESTING YOUR MODEL:

Test prompt 1 (Chest Pain):
Chest pain

Patient is a 65 year old male with history of HTN, diabetes who presents with 
chest pain. Pain started 2 hours ago while at rest, 7/10 severity. 

Labs: Troponin-T 0.55

─────────────────────────────────────────────────────────────────────────────────

Test prompt 2 (Shortness of Breath):
Shortness of breath

58 year old female with CHF presents with worsening dyspnea x 3 days. Reports 
orthopnea, bilateral leg swelling.

Physical exam: JVD present, bilateral crackles, 2+ pitting edema

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 MODEL SPECIFICATIONS:
   • Base: Qwen 2.5 1.5B
   • Fine-tuned: Medical diagnostic reasoning
   • Format: GGUF (Ollama-compatible)
   • Size: {os.path.getsize(gguf_path) / (1024**2):.2f} MB
   • Quantization: Optimized for mobile devices

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 TIPS:
   • For Android, Q4_K_M quantization is recommended (good balance of size/quality)
   • Ensure your device has at least 2GB free RAM for smooth inference
   • Use the model name '{model_name}' when calling via API
   • Adjust temperature (0.1-1.0) for more/less creative responses

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔗 USEFUL LINKS:
   • Ollama Docs: https://github.com/ollama/ollama/blob/main/docs
   • Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md
   • Android App: https://github.com/ollama/ollama-android
   • Model Library: https://ollama.ai/library

╚════════════════════════════════════════════════════════════════════════════════╝
"""
    
    print(instructions)
    
    # Save to file
    instructions_file = f"OLLAMA_SETUP_{model_name}.txt"
    with open(instructions_file, 'w') as f:
        f.write(instructions)
    
    print(f"\n💾 Instructions saved to: {instructions_file}")
    
    return instructions_file

def main():
    """Main conversion pipeline"""
    print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║          FINE-TUNED MODEL TO OLLAMA CONVERTER                                  ║
║          Convert your clinical reasoning model for Android/Desktop             ║
╚════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Step 0: Check requirements
        check_requirements()
        
        # Get user inputs
        print("\n" + "="*80)
        print("CONFIGURATION")
        print("="*80)
        
        adapter_path = input("\nEnter LoRA adapter path [./qwen1.5b-clinical-reasoning/adapters]: ").strip()
        adapter_path = adapter_path or "./qwen1.5b-clinical-reasoning/adapters"
        
        if not os.path.exists(adapter_path):
            print(f"❌ Adapter path not found: {adapter_path}")
            sys.exit(1)
        
        output_name = input("Enter output model name [qwen-clinical]: ").strip() or "qwen-clinical"
        
        # Step 1: Merge LoRA weights
        merged_path = merge_lora_weights(adapter_path=adapter_path, output_path=f"./{output_name}-merged")
        
        # Step 2: Convert to GGUF
        gguf_path = convert_to_gguf(merged_path, output_name=output_name)
        
        if gguf_path is None:
            print("\n⚠️  GGUF conversion skipped or failed.")
            print("You can manually convert using llama.cpp later.")
            print(f"Merged model is available at: {merged_path}")
            sys.exit(0)
        
        # Step 3: Create Modelfile
        modelfile_path = create_modelfile(gguf_path, output_name=output_name)
        
        # Step 4: Create setup instructions
        create_ollama_instructions(gguf_path, modelfile_path, model_name=output_name)
        
        print("\n" + "="*80)
        print("✅ CONVERSION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📦 Your files are ready:")
        print(f"   • GGUF Model: {os.path.abspath(gguf_path)}")
        print(f"   • Modelfile: {os.path.abspath(modelfile_path)}")
        print(f"\n🚀 Next step: Import into Ollama (see instructions above)")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Conversion cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()