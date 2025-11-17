import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_final_model():
    """Load the final model with both adapters"""
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load both adapters
    print("Loading both adapters...")
    model = PeftModel.from_pretrained(
        base_model,
        "./qwen1.5b-clinical-reasoning/final_adapters"
    )
    
    return model, tokenizer

def inference():
    model, tokenizer = load_final_model()
    
    print("="*80)
    print("MEDICAL DIAGNOSTIC ASSISTANT INFERENCE")
    print("="*80)
    print("The model will analyze clinical notes and provide:")
    print("1. Most suspected disease")
    print("2. Other diseases with significant risk")
    print("3. Diagnostic reasoning") 
    print("4. Precautions for the most suspected disease")
    print("\nType 'quit' to exit\n")
    
    while True:
        clinical_note = input("Enter clinical note: ").strip()
        
        if clinical_note.lower() in ['quit', 'exit', 'q']:
            break
        
        # Create comprehensive prompt
        instruction = """Analyze this clinical note and provide:
1. Most suspected disease
2. Other diseases with significant risk  
3. Diagnostic reasoning
4. Precautions for the most suspected disease

Please structure your response as follows:
**Most Suspected Disease:** [disease name]
**Other Significant Risks:** [list of other diseases]
**Diagnostic Reasoning:** [your reasoning chain]
**Precautions:** [precautions for the most suspected disease]"""
        
        prompt = f"<|im_start|>user\n{instruction}\n\nClinical Note:\n{clinical_note}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        
        # MOVE INPUTS TO GPU - THIS IS THE FIX
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "assistant" in response:
            assistant_part = response.split("assistant")[-1].strip()
        else:
            assistant_part = response
        
        print(f"\nAssistant Response:")
        print("-" * 50)
        print(assistant_part)
        print("-" * 50)
        print()

if __name__ == "__main__":
    inference()