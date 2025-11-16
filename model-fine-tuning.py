import json
import os
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import numpy as np
from sklearn.model_selection import train_test_split
import gc

def extract_reasoning_chains_from_tree(tree_node, current_path=None):
    """Extract all reasoning paths from the diagnostic tree"""
    if current_path is None:
        current_path = []
    
    reasoning_chains = []
    
    for key, value in tree_node.items():
        # Clean the key (remove $Cause_1, $Intermedia_3, etc.)
        clean_key = key.split('$')[0].strip()
        new_path = current_path + [clean_key]
        
        if isinstance(value, dict) and value:
            # Continue traversing the tree
            reasoning_chains.extend(extract_reasoning_chains_from_tree(value, new_path))
        else:
            # Reached the end of a reasoning chain
            reasoning_chains.append(new_path)
    
    return reasoning_chains

def load_direct_dataset(json_folder_path):
    """Load the actual MIMIC-IV-Ext-DiReCT dataset"""
    data = []
    
    # Traverse through disease folders
    for root, dirs, files in os.walk(json_folder_path):
        for file in files:
            if file.endswith('.json') and not file.startswith('.'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        disease_data = json.load(f)
                    
                    # Get disease name from folder structure
                    disease_name = os.path.basename(root)
                    if disease_name == os.path.basename(json_folder_path):
                        disease_name = os.path.basename(file).replace('.json', '')
                    
                    # Extract inputs (clinical notes)
                    clinical_note = ""
                    for i in range(1, 10):  # Check input1 to input9
                        input_key = f"input{i}"
                        if input_key in disease_data:
                            clinical_note += disease_data[input_key] + "\n\n"
                    
                    # Skip if no clinical note found
                    if not clinical_note.strip():
                        continue
                    
                    # Extract reasoning chains from the tree structure
                    reasoning_chains = []
                    for key, value in disease_data.items():
                        if not key.startswith('input') and isinstance(value, dict):
                            chains = extract_reasoning_chains_from_tree({key: value})
                            reasoning_chains.extend(chains)
                    
                    # Create training examples from each reasoning chain
                    for chain in reasoning_chains:
                        if len(chain) >= 2:  # Only use chains with at least 2 steps
                            reasoning_text = " → ".join(chain)
                            
                            data.append({
                                "disease": disease_name,
                                "clinical_note": clinical_note.strip(),
                                "reasoning_chain": reasoning_text
                            })
                            
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
    
    return pd.DataFrame(data)

def load_model_and_tokenizer():
    """Load Qwen model with 4-bit quantization for memory efficiency"""
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )
    
    # Set padding token to eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Configure 4-bit quantization for maximum memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,  # Double quantization for even more memory savings
        bnb_4bit_quant_type="nf4",  # Normal Float 4-bit quantization
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    )
    
    print("Loading model in 4-bit quantization mode for memory efficiency...")
    print(f"Expected memory usage: ~1.5GB instead of ~6GB")
    
    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def setup_lora(model):
    """Configure LoRA for efficient fine-tuning on quantized model"""
    # Enable gradient checkpointing with recommended settings
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    # Prepare model for k-bit training (works with 4-bit quantization)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True
    )
    
    # LoRA configuration - optimized for memory efficiency
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # Reduced rank for lower memory usage
        lora_alpha=16,  # Scaled accordingly
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Reduced targets for memory
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def preprocess_function(examples, tokenizer):
    """Preprocess data for training"""
    texts = []
    
    for i in range(len(examples['clinical_note'])):
        # Create the full text with instruction
        instruction = "Analyze this clinical note and provide diagnostic reasoning:"
        full_text = f"<|im_start|>user\n{instruction}\n\nClinical Note:\n{examples['clinical_note'][i]}<|im_end|>\n<|im_start|>assistant\n{examples['reasoning_chain'][i]}<|im_end|>"
        texts.append(full_text)
    
    # Tokenize with proper settings
    model_inputs = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=1024,  # Reduced from 2048 for memory efficiency
        return_tensors=None,
    )
    
    # Create labels - copy input_ids and mask the instruction part
    labels = []
    for i, text in enumerate(texts):
        input_ids = model_inputs["input_ids"][i].copy()
        label = input_ids.copy()
        
        # Find where assistant response starts
        assistant_marker = "<|im_start|>assistant\n"
        assistant_start_pos = text.find(assistant_marker)
        
        if assistant_start_pos != -1:
            # Tokenize the prefix to find the position in tokens
            prefix_text = text[:assistant_start_pos + len(assistant_marker)]
            prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
            prefix_len = len(prefix_tokens)
            
            # Mask everything before assistant response with -100
            for j in range(prefix_len):
                if j < len(label):
                    label[j] = -100
        
        labels.append(label)
    
    model_inputs["labels"] = labels
    
    return model_inputs

def print_dataset_samples(dataset, dataset_name, num_samples=3):
    """Print sample examples from a dataset"""
    print("\n" + "="*80)
    print(f"{dataset_name.upper()} SAMPLES")
    print("="*80)
    
    for i in range(min(num_samples, len(dataset))):
        example = dataset[i]
        print(f"\n--- Sample {i+1} ---")
        print(f"Disease: {example['disease']}")
        print(f"\nClinical Note (first 200 chars):\n{example['clinical_note'][:200]}...")
        print(f"\nReasoning Chain:\n{example['reasoning_chain']}")
        print("-" * 80)

def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(f"  Free: {(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / 1024**3:.2f} GB")

def evaluate_model(model, tokenizer, original_dataset, num_samples=3):
    """Evaluate model on sample data"""
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    model.eval()
    
    for i in range(min(num_samples, len(original_dataset))):
        example = original_dataset[i]
        
        # Create prompt (user part only)
        instruction = "Analyze this clinical note and provide diagnostic reasoning:"
        input_text = f"<|im_start|>user\n{instruction}\n\nClinical Note:\n{example['clinical_note']}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,  # Reduced for memory
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "assistant" in generated_text:
            assistant_index = generated_text.rfind("assistant")
            if assistant_index != -1:
                assistant_response = generated_text[assistant_index + 9:].strip()
            else:
                assistant_response = generated_text
        else:
            assistant_response = generated_text
        
        print(f"\n--- Example {i+1} | Disease: {example['disease']} ---")
        print(f"CLINICAL NOTE: {example['clinical_note'][:150]}...")
        print(f"EXPECTED REASONING: {example['reasoning_chain']}")
        print(f"GENERATED REASONING: {assistant_response}")
        print("-" * 80)
    
    model.train()

def main():
    # Clear GPU cache at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print("="*80)
    print("GPU MEMORY STATUS AT START")
    print("="*80)
    print_gpu_memory()
    
    # Load the actual DiReCT dataset
    print("\n" + "="*80)
    print("Loading MIMIC-IV-Ext-DiReCT dataset...")
    print("="*80)
    json_folder_path = "mimic-iv-ext-direct-1.0.0"
    
    df = load_direct_dataset(json_folder_path)
    
    if len(df) == 0:
        print("No data found! Check your folder path and JSON structure.")
        return
    
    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}")
    print(f"Total examples loaded: {len(df)}")
    print(f"Unique diseases: {df['disease'].nunique()}")
    print(f"\nDisease distribution:")
    disease_counts = df['disease'].value_counts()
    for disease, count in disease_counts.head(10).items():
        print(f"  {disease}: {count}")
    if len(disease_counts) > 10:
        print(f"  ... and {len(disease_counts) - 10} more diseases")
    
    # Split data 80/20
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['disease'])
    
    print(f"\n{'='*80}")
    print("TRAIN/TEST SPLIT")
    print(f"{'='*80}")
    print(f"Training set size: {len(train_df)} examples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation set size: {len(eval_df)} examples ({len(eval_df)/len(df)*100:.1f}%)")
    print(f"Training diseases: {train_df['disease'].nunique()}")
    print(f"Validation diseases: {eval_df['disease'].nunique()}")
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    eval_dataset = Dataset.from_pandas(eval_df.reset_index(drop=True))
    
    # Print samples from both datasets BEFORE training
    print_dataset_samples(train_dataset, "TRAINING SET", num_samples=3)
    print_dataset_samples(eval_dataset, "VALIDATION SET", num_samples=3)
    
    # Keep original datasets for evaluation
    original_train_dataset = train_dataset
    original_eval_dataset = eval_dataset
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": eval_dataset
    })
    
    # Load model with 4-bit quantization
    print(f"\n{'='*80}")
    print("Loading Qwen 1.5B model with 4-bit quantization...")
    print(f"{'='*80}")
    model, tokenizer = load_model_and_tokenizer()
    
    print("\n" + "="*80)
    print("GPU MEMORY AFTER MODEL LOADING")
    print("="*80)
    print_gpu_memory()
    
    # Preprocess data
    print("\nPreprocessing datasets...")
    
    def preprocess_batch(examples):
        return preprocess_function(examples, tokenizer)
    
    tokenized_datasets = dataset_dict.map(
        preprocess_batch,
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
        desc="Tokenizing datasets",
    )
    
    print(f"Tokenized training examples: {len(tokenized_datasets['train'])}")
    print(f"Tokenized validation examples: {len(tokenized_datasets['validation'])}")
    
    # Baseline evaluation before training
    print(f"\n{'='*80}")
    print("BASELINE EVALUATION (Before Training)")
    print(f"{'='*80}")
    
    evaluate_model(model, tokenizer, original_eval_dataset)
    
    # Setup LoRA fine-tuning
    print(f"\n{'='*80}")
    print("Setting up LoRA...")
    print(f"{'='*80}")
    model = setup_lora(model)
    
    print("\n" + "="*80)
    print("GPU MEMORY AFTER LoRA SETUP")
    print("="*80)
    print_gpu_memory()
    
    # Training arguments optimized for low memory
    training_args = TrainingArguments(
        output_dir="./qwen1.5b-clinical-reasoning",
        per_device_train_batch_size=2,  # Minimal batch size for 6GB GPU
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size = 8
        learning_rate=2e-5,
        num_train_epochs=2,
        logging_dir="./logs",
        logging_steps=10,
        eval_steps=100,
        save_steps=200,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=50,
        fp16=True,  # Use FP16 for memory efficiency
        dataloader_pin_memory=False,  # Disable to save memory
        save_total_limit=2,
        remove_unused_columns=True,
        report_to="none",
        gradient_checkpointing=False,
        optim="paged_adamw_8bit",  # 8-bit optimizer for memory efficiency
        max_grad_norm=0.3,
    )
    
    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    
    # Train
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Total training steps: {len(tokenized_datasets['train']) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")
    
    train_result = trainer.train()
    
    # Final evaluation
    print(f"\n{'='*80}")
    print("FINAL EVALUATION (After Training)")
    print(f"{'='*80}")
    final_metrics = trainer.evaluate()
    print(f"Final validation loss: {final_metrics['eval_loss']:.4f}")
    
    evaluate_model(model, tokenizer, original_eval_dataset)
    
    # Save model
    print(f"\n{'='*80}")
    print("Saving fine-tuned model...")
    print(f"{'='*80}")
    trainer.save_model()
    tokenizer.save_pretrained("./qwen1.5b-clinical-reasoning")
    
    # Save LoRA adapters separately
    model.save_pretrained("./qwen1.5b-clinical-reasoning/adapters")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print("Model saved in: ./qwen1.5b-clinical-reasoning")
    print("Adapters saved in: ./qwen1.5b-clinical-reasoning/adapters")
    print(f"Total training examples: {len(train_df)}")
    print(f"Total validation examples: {len(eval_df)}")
    print(f"Training diseases: {train_df['disease'].nunique()}")
    
    print("\n" + "="*80)
    print("FINAL GPU MEMORY STATUS")
    print("="*80)
    print_gpu_memory()

if __name__ == "__main__":
    main()