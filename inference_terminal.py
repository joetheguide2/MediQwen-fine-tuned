"""
Terminal Chat Interface for Fine-tuned Clinical Reasoning Model
Run this script to interact with your fine-tuned model in the terminal

Usage:
    python inference_terminal.py

Requirements:
    - transformers
    - torch
    - peft
    - colorama
    - bitsandbytes

The script will automatically load your fine-tuned model from:
    ./qwen1.5b-clinical-reasoning/adapters/
    ./qwen1.5b-clinical-reasoning/ (tokenizer and config)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import sys
import os
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init(autoreset=True)

def print_colored(text, color=Fore.WHITE, style=Style.NORMAL):
    """Print colored text to terminal"""
    print(f"{style}{color}{text}{Style.RESET_ALL}")

def load_finetuned_model(adapter_path="./qwen1.5b-clinical-reasoning/adapters"):
    """Load the fine-tuned model with 4-bit quantization"""
    print_colored("\n" + "="*80, Fore.CYAN, Style.BRIGHT)
    print_colored("Loading Fine-tuned Clinical Reasoning Model...", Fore.CYAN, Style.BRIGHT)
    print_colored("="*80, Fore.CYAN, Style.BRIGHT)
    
    # Check if mobile config exists
    mobile_config_path = "./qwen1.5b-clinical-reasoning/mobile_config.json"
    mobile_config = None
    if os.path.exists(mobile_config_path):
        import json
        with open(mobile_config_path, 'r') as f:
            mobile_config = json.load(f)
        print_colored(f"📱 Mobile optimization detected: {mobile_config['mode'].upper()}", Fore.YELLOW)
        print_colored(f"   LoRA rank: r={mobile_config['lora_r']}", Fore.YELLOW)
        print_colored(f"   Max sequence: {mobile_config['max_seq_length']} tokens", Fore.YELLOW)
        print_colored(f"   Expected mobile size: ~{mobile_config['expected_mobile_size_mb']}MB\n", Fore.YELLOW)
    
    # IMPORTANT: Load tokenizer FIRST to check vocabulary size
    print_colored("📝 Loading tokenizer first...", Fore.YELLOW)
    tokenizer = AutoTokenizer.from_pretrained(
        "./qwen1.5b-clinical-reasoning",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print_colored(f"   Tokenizer vocabulary size: {len(tokenizer)}", Fore.CYAN)
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    )
    
    print_colored("📦 Loading base model (Qwen 2.5 1.5B) with 4-bit quantization...", Fore.YELLOW)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # CRITICAL FIX: Resize model embeddings to match tokenizer
    print_colored(f"🔧 Resizing model embeddings to match tokenizer ({len(tokenizer)} tokens)...", Fore.YELLOW)
    base_model.resize_token_embeddings(len(tokenizer))
    
    print_colored("🔧 Loading fine-tuned LoRA adapters...", Fore.YELLOW)
    
    # Load LoRA adapters
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print_colored("✅ Successfully loaded fine-tuned adapters!", Fore.GREEN, Style.BRIGHT)
    except Exception as e:
        print_colored(f"❌ Error loading adapters: {e}", Fore.RED, Style.BRIGHT)
        print_colored("📝 Using base model without fine-tuning...", Fore.YELLOW)
        model = base_model
    
    model.eval()
    
    # Print memory usage
    if torch.cuda.is_available():
        print_colored(f"\n💾 GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated", Fore.CYAN)
    
    print_colored("✅ Model ready for inference!\n", Fore.GREEN, Style.BRIGHT)
    
    return model, tokenizer, mobile_config

def generate_response(model, tokenizer, clinical_note, mobile_config=None, max_new_tokens=200, temperature=0.7):
    """Generate diagnostic reasoning from clinical note"""
    
    # Adjust max_new_tokens based on mobile config
    if mobile_config:
        if mobile_config['mode'] == 'extreme':
            max_new_tokens = min(max_new_tokens, 100)  # Shorter for extreme mode
        elif mobile_config['mode'] == 'ultra_light':
            max_new_tokens = min(max_new_tokens, 150)  # Medium for ultra-light
    
    # Format input
    instruction = "Analyze this clinical note and provide diagnostic reasoning:"
    prompt = f"<|im_start|>user\n{instruction}\n\nClinical Note:\n{clinical_note}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize with appropriate max_length
    max_input_length = mobile_config['max_seq_length'] if mobile_config else 1024
    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_input_length, truncation=True)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant's response
    if "assistant" in generated_text:
        assistant_index = generated_text.rfind("assistant")
        if assistant_index != -1:
            response = generated_text[assistant_index + 9:].strip()
        else:
            response = generated_text
    else:
        response = generated_text
    
    return response

def print_help():
    """Print help menu"""
    print_colored("\n" + "="*80, Fore.CYAN)
    print_colored("AVAILABLE COMMANDS", Fore.CYAN, Style.BRIGHT)
    print_colored("="*80, Fore.CYAN)
    print_colored("  /help    - Show this help menu", Fore.WHITE)
    print_colored("  /clear   - Clear the screen", Fore.WHITE)
    print_colored("  /example - Show example clinical notes", Fore.WHITE)
    print_colored("  /quit    - Exit the program", Fore.WHITE)
    print_colored("  /exit    - Exit the program", Fore.WHITE)
    print_colored("\nTo analyze a clinical note, just paste it and press Enter twice (blank line)", Fore.YELLOW)
    print_colored("="*80 + "\n", Fore.CYAN)

def show_examples():
    """Show example clinical notes"""
    print_colored("\n" + "="*80, Fore.CYAN)
    print_colored("EXAMPLE CLINICAL NOTES", Fore.CYAN, Style.BRIGHT)
    print_colored("="*80, Fore.CYAN)
    
    examples = [
        {
            "title": "Chest Pain (NSTEMI)",
            "note": """Chest pain

Patient is a 65 year old male with history of HTN, diabetes, hyperlipidemia who presents with chest pain. Pain started 2 hours ago while at rest, described as pressure-like, 7/10 severity. Associated with shortness of breath and diaphoresis. No relief with rest.

Labs: Troponin-T 0.55, CK-MB elevated, normal renal function"""
        },
        {
            "title": "Altered Mental Status",
            "note": """Altered mental status

77 year old man with dementia, HTN who presents with confusion x 2 days. Family reports patient more lethargic than usual, not eating well. No fever reported.

Vitals: BP 145/85, HR 88, Temp 98.2F
Neuro: Oriented to person only, decreased attention span"""
        },
        {
            "title": "Shortness of Breath",
            "note": """Shortness of breath

58 year old female with history of CHF presents with worsening dyspnea x 3 days. Reports orthopnea, PND, bilateral leg swelling. Has not been compliant with medications.

Physical exam: JVD present, bilateral crackles on lung exam, 2+ pitting edema bilateral lower extremities"""
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print_colored(f"\n{i}. {example['title']}", Fore.GREEN, Style.BRIGHT)
        print_colored("-" * 40, Fore.WHITE)
        print_colored(example['note'], Fore.WHITE)
        print_colored("-" * 40, Fore.WHITE)
    
    print_colored("\nCopy and paste any of these examples to test the model!\n", Fore.YELLOW)

def chat_loop(model, tokenizer, mobile_config=None):
    """Main chat loop"""
    print_colored("\n" + "="*80, Fore.GREEN, Style.BRIGHT)
    print_colored("🏥 CLINICAL REASONING ASSISTANT", Fore.GREEN, Style.BRIGHT)
    print_colored("="*80, Fore.GREEN, Style.BRIGHT)
    print_colored("\nWelcome! I can analyze clinical notes and provide diagnostic reasoning.", Fore.WHITE)
    print_colored("Type /help for available commands or paste a clinical note to begin.\n", Fore.YELLOW)
    
    if mobile_config:
        print_colored(f"📱 Running in {mobile_config['mode'].upper()} mode", Fore.CYAN)
        print_colored(f"   (optimized for {mobile_config['expected_mobile_size_mb']}MB mobile deployment)\n", Fore.CYAN)
    
    while True:
        try:
            # Get user input
            print_colored("═" * 80, Fore.BLUE)
            print_colored("📋 Enter clinical note (press Enter twice when done):", Fore.BLUE, Style.BRIGHT)
            print_colored("─" * 80, Fore.BLUE)
            
            lines = []
            empty_count = 0
            
            while True:
                line = input()
                
                # Check for commands
                if line.strip().startswith('/'):
                    command = line.strip().lower()
                    
                    if command in ['/quit', '/exit']:
                        print_colored("\n👋 Thank you for using Clinical Reasoning Assistant!", Fore.GREEN, Style.BRIGHT)
                        print_colored("Goodbye!\n", Fore.GREEN)
                        return
                    
                    elif command == '/help':
                        print_help()
                        break
                    
                    elif command == '/clear':
                        import os
                        os.system('clear' if os.name == 'posix' else 'cls')
                        print_colored("🏥 CLINICAL REASONING ASSISTANT\n", Fore.GREEN, Style.BRIGHT)
                        break
                    
                    elif command == '/example':
                        show_examples()
                        break
                    
                    else:
                        print_colored(f"❌ Unknown command: {command}", Fore.RED)
                        print_colored("Type /help for available commands\n", Fore.YELLOW)
                        break
                
                # Regular input processing
                if line.strip() == '':
                    empty_count += 1
                    if empty_count >= 2 or (len(lines) > 0 and empty_count >= 1):
                        break
                else:
                    empty_count = 0
                    lines.append(line)
            
            clinical_note = '\n'.join(lines).strip()
            
            # Skip if no note provided
            if not clinical_note or clinical_note.startswith('/'):
                continue
            
            # Generate response
            print_colored("\n🤔 Analyzing clinical note...\n", Fore.YELLOW, Style.BRIGHT)
            
            response = generate_response(model, tokenizer, clinical_note, mobile_config=mobile_config)
            
            # Display response
            print_colored("═" * 80, Fore.GREEN)
            print_colored("💡 DIAGNOSTIC REASONING:", Fore.GREEN, Style.BRIGHT)
            print_colored("─" * 80, Fore.GREEN)
            print_colored(response, Fore.WHITE, Style.BRIGHT)
            print_colored("═" * 80 + "\n", Fore.GREEN)
            
        except KeyboardInterrupt:
            print_colored("\n\n👋 Interrupted. Type /quit to exit or continue...\n", Fore.YELLOW)
            continue
        except Exception as e:
            print_colored(f"\n❌ Error: {e}\n", Fore.RED, Style.BRIGHT)
            continue

def main():
    """Main function"""
    try:
        # Load model
        model, tokenizer, mobile_config = load_finetuned_model()
        
        # Start chat loop
        chat_loop(model, tokenizer, mobile_config)
        
    except Exception as e:
        print_colored(f"\n❌ Fatal error: {e}\n", Fore.RED, Style.BRIGHT)
        print_colored("Please check that your model files exist in:", Fore.YELLOW)
        print_colored("  - ./qwen1.5b-clinical-reasoning/adapters/", Fore.WHITE)
        print_colored("  - ./qwen1.5b-clinical-reasoning/ (tokenizer)", Fore.WHITE)
        sys.exit(1)

if __name__ == "__main__":
    # Install colorama if not available
    try:
        import colorama
    except ImportError:
        print("Installing colorama for colored terminal output...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "colorama"])
        from colorama import init, Fore, Style
        init(autoreset=True)
    
    main()