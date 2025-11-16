"""
Base Qwen 1.5B Inference - NO Fine-tuning
Compare this with your fine-tuned model to see the improvement

Usage:
    python base_model_inference.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init(autoreset=True)

def print_colored(text, color=Fore.WHITE, style=Style.NORMAL):
    """Print colored text to terminal"""
    print(f"{style}{color}{text}{Style.RESET_ALL}")

def load_base_model():
    """Load BASE Qwen model without any fine-tuning"""
    print_colored("\n" + "="*80, Fore.CYAN, Style.BRIGHT)
    print_colored("Loading BASE Qwen 2.5 1.5B Model (No Fine-tuning)", Fore.CYAN, Style.BRIGHT)
    print_colored("="*80, Fore.CYAN, Style.BRIGHT)
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    )
    
    print_colored("📦 Loading base model (Qwen 2.5 1.5B Instruct)...", Fore.YELLOW)
    
    # Load base model WITHOUT any adapters
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    print_colored("📝 Loading tokenizer...", Fore.YELLOW)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    
    # Print memory usage
    if torch.cuda.is_available():
        print_colored(f"\n💾 GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated", Fore.CYAN)
    
    print_colored("✅ Base model ready!\n", Fore.GREEN, Style.BRIGHT)
    print_colored("⚠️  NOTE: This is the UNTUNED base model", Fore.YELLOW, Style.BRIGHT)
    print_colored("   Compare outputs with your fine-tuned model to see improvement!\n", Fore.YELLOW)
    
    return model, tokenizer

def generate_response(model, tokenizer, clinical_note, max_new_tokens=200, temperature=0.7):
    """Generate diagnostic reasoning from clinical note using BASE model"""
    
    # Use SAME prompt format as fine-tuned model for fair comparison
    instruction = "Analyze this clinical note and provide diagnostic reasoning:"
    prompt = f"<|im_start|>user\n{instruction}\n\nClinical Note:\n{clinical_note}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    
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
    
    print_colored("\nCopy and paste any of these examples to test the BASE model!\n", Fore.YELLOW)

def chat_loop(model, tokenizer):
    """Main chat loop"""
    print_colored("\n" + "="*80, Fore.RED, Style.BRIGHT)
    print_colored("🔬 BASE QWEN MODEL (NO FINE-TUNING)", Fore.RED, Style.BRIGHT)
    print_colored("="*80, Fore.RED, Style.BRIGHT)
    print_colored("\n⚠️  This is the UNTUNED base model for comparison", Fore.YELLOW, Style.BRIGHT)
    print_colored("   Responses will be generic and not specialized for clinical reasoning", Fore.YELLOW)
    print_colored("\nType /help for available commands or paste a clinical note to begin.\n", Fore.WHITE)
    
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
                        print_colored("\n👋 Thank you for testing the base model!", Fore.GREEN, Style.BRIGHT)
                        print_colored("Now try inference_terminal.py to see your fine-tuned model!\n", Fore.GREEN)
                        return
                    
                    elif command == '/help':
                        print_help()
                        break
                    
                    elif command == '/clear':
                        import os
                        os.system('clear' if os.name == 'posix' else 'cls')
                        print_colored("🔬 BASE QWEN MODEL (NO FINE-TUNING)\n", Fore.RED, Style.BRIGHT)
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
            print_colored("\n🤔 Analyzing with BASE model (no fine-tuning)...\n", Fore.YELLOW, Style.BRIGHT)
            
            response = generate_response(model, tokenizer, clinical_note)
            
            # Display response
            print_colored("═" * 80, Fore.RED)
            print_colored("💡 BASE MODEL OUTPUT:", Fore.RED, Style.BRIGHT)
            print_colored("─" * 80, Fore.RED)
            print_colored(response, Fore.WHITE, Style.BRIGHT)
            print_colored("═" * 80 + "\n", Fore.RED)
            
            print_colored("💭 TIP: Run inference_terminal.py to compare with fine-tuned model", Fore.CYAN)
            
        except KeyboardInterrupt:
            print_colored("\n\n👋 Interrupted. Type /quit to exit or continue...\n", Fore.YELLOW)
            continue
        except Exception as e:
            print_colored(f"\n❌ Error: {e}\n", Fore.RED, Style.BRIGHT)
            continue

def main():
    """Main function"""
    try:
        # Load base model (no fine-tuning)
        model, tokenizer = load_base_model()
        
        # Start chat loop
        chat_loop(model, tokenizer)
        
    except Exception as e:
        print_colored(f"\n❌ Fatal error: {e}\n", Fore.RED, Style.BRIGHT)
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