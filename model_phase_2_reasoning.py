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
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel
import numpy as np
from sklearn.model_selection import train_test_split
import gc

def extract_reasoning_chains_from_tree(tree_node, current_path=None):
    """Extract all reasoning paths from the diagnostic tree"""
    if current_path is None:
        current_path = []
    
    reasoning_chains = []
    
    for key, value in tree_node.items():
        clean_key = key.split('$')[0].strip()
        new_path = current_path + [clean_key]
        
        if isinstance(value, dict) and value:
            reasoning_chains.extend(extract_reasoning_chains_from_tree(value, new_path))
        else:
            reasoning_chains.append(new_path)
    
    return reasoning_chains

def load_direct_dataset(json_folder_path):
    """Load the actual MIMIC-IV-Ext-DiReCT dataset"""
    data = []
    
    for root, dirs, files in os.walk(json_folder_path):
        for file in files:
            if file.endswith('.json') and not file.startswith('.'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        disease_data = json.load(f)
                    
                    disease_name = os.path.basename(root)
                    if disease_name == os.path.basename(json_folder_path):
                        disease_name = os.path.basename(file).replace('.json', '')
                    
                    clinical_note = ""
                    for i in range(1, 10):
                        input_key = f"input{i}"
                        if input_key in disease_data:
                            clinical_note += disease_data[input_key] + "\n\n"
                    
                    if not clinical_note.strip():
                        continue
                    
                    reasoning_chains = []
                    for key, value in disease_data.items():
                        if not key.startswith('input') and isinstance(value, dict):
                            chains = extract_reasoning_chains_from_tree({key: value})
                            reasoning_chains.extend(chains)
                    
                    for chain in reasoning_chains:
                        if len(chain) >= 2:
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

def load_model_with_frozen_adapters():
    """Load model with frozen phase 1 adapters using the SAME tokenizer from phase 1"""
    
    # Load the tokenizer from phase 1 to ensure consistency
    print("Loading tokenizer from phase 1...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "./qwen1.5b-symptoms-precautions",
            trust_remote_code=True,
            use_fast=True
        )
        print("Successfully loaded tokenizer from phase 1")
    except:
        print("Failed to load tokenizer from phase 1, using base model tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            trust_remote_code=True,
            use_fast=True
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model with quantization - use the same config as phase 1
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Use float16 instead of bfloat16 for compatibility
    )
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # First, resize the model to match the tokenizer
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Now load phase 1 adapters and freeze them
    print("Loading and freezing phase 1 adapters...")
    try:
        model = PeftModel.from_pretrained(
            base_model,
            "./qwen1.5b-symptoms-precautions/phase1_adapters",
            is_trainable=False  # Freeze phase 1 adapters
        )
        print("Successfully loaded phase 1 adapters")
    except Exception as e:
        print(f"Error loading phase 1 adapters: {e}")
        print("Continuing without phase 1 adapters...")
        model = base_model
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # Add new LoRA adapters for phase 2 with different target modules
    lora_config_phase2 = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,  # Reduced from 8 for memory
        lora_alpha=8,  # Reduced from 16
        lora_dropout=0.05,
        target_modules=["gate_proj", "up_proj"],  # Reduced target modules
        bias="none",
    )
    
    model = get_peft_model(model, lora_config_phase2)
    
    print("\nTrainable parameters (Phase 2 only):")
    model.print_trainable_parameters()
    
    return model, tokenizer

class CustomDataCollator:
    """Custom data collator that handles padding properly for variable length sequences"""
    def __init__(self, tokenizer, max_length=512):  # Reduced from 1024
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, features):
        # Separate input_ids, attention_mask, and labels
        input_ids = [feature['input_ids'] for feature in features]
        labels = [feature['labels'] for feature in features]
        
        # Find max length in this batch
        batch_max_length = max(len(seq) for seq in input_ids)
        batch_max_length = min(batch_max_length, self.max_length)
        
        # Pad sequences
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for i in range(len(input_ids)):
            input_seq = input_ids[i]
            label_seq = labels[i]
            
            # Truncate if too long
            if len(input_seq) > batch_max_length:
                input_seq = input_seq[:batch_max_length]
                label_seq = label_seq[:batch_max_length]
            
            # Pad sequences
            pad_length = batch_max_length - len(input_seq)
            
            padded_input = input_seq + [self.tokenizer.pad_token_id] * pad_length
            padded_attention = [1] * len(input_seq) + [0] * pad_length
            padded_label = label_seq + [-100] * pad_length  # Use -100 for padding in labels
            
            padded_input_ids.append(padded_input)
            padded_attention_mask.append(padded_attention)
            padded_labels.append(padded_label)
        
        # Convert to tensors
        batch = {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long),
        }
        
        return batch

def preprocess_reasoning_function(examples, tokenizer):
    """Preprocess reasoning data with new comprehensive prompt format and TRUNCATION"""
    texts = []
    
    for i in range(len(examples['clinical_note'])):
        clinical_note = examples['clinical_note'][i]
        reasoning_chain = examples['reasoning_chain'][i]
        disease = examples['disease'][i]
        
        # Truncate clinical note to save memory
        clinical_note = clinical_note[:1500]  # Limit to 1500 chars
        
        # New comprehensive prompt format
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
        
        # Create the target response
        target_response = f"""**Most Suspected Disease:** {disease}
**Other Significant Risks:** [Related conditions based on symptoms]
**Diagnostic Reasoning:** {reasoning_chain}
**Precautions:** [Standard precautions for {disease} including medication adherence, follow-up visits, and symptom monitoring]"""
        
        full_text = f"<|im_start|>user\n{instruction}\n\nClinical Note:\n{clinical_note}<|im_end|>\n<|im_start|>assistant\n{target_response}<|im_end|>"
        texts.append(full_text)
    
    # Tokenize WITH truncation to prevent OOM
    tokenized = tokenizer(
        texts,
        truncation=True,  # Enable truncation
        padding=False,
        max_length=512,   # Strict max length
        return_tensors=None,
        add_special_tokens=True,
    )
    
    # Create labels - for causal LM, labels are the same as input_ids
    result = {
        'input_ids': tokenized['input_ids'],
        'labels': tokenized['input_ids'].copy()  # Copy for labels
    }
    
    return result

def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")

def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print("="*80)
    print("PHASE 2: CLINICAL REASONING TRAINING (MEMORY OPTIMIZED)")
    print("="*80)
    
    # Check if phase 1 adapters exist
    if not os.path.exists("./qwen1.5b-symptoms-precautions/phase1_adapters"):
        print("WARNING: Phase 1 adapters not found!")
        print("Please run phase 1 training first.")
        print("Continuing without phase 1 adapters...")
    
    # Load reasoning dataset
    print("\nLoading reasoning dataset...")
    json_folder_path = "mimic-iv-ext-direct-1.0.0"
    df = load_direct_dataset(json_folder_path)
    
    if len(df) == 0:
        print("No reasoning data found! Check your folder path.")
        return
    
    print(f"Reasoning examples loaded: {len(df)}")
    print(f"Unique diseases: {df['disease'].nunique()}")
    
    # Show sample of the data structure
    print(f"\nSample data structure:")
    for i in range(min(2, len(df))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Disease: {df.iloc[i]['disease']}")
        print(f"Clinical Note (first 200 chars): {df.iloc[i]['clinical_note'][:200]}...")
        print(f"Reasoning Chain: {df.iloc[i]['reasoning_chain']}")
    
    # Split data
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['disease'])
    
    print(f"\nTraining set: {len(train_df)} examples")
    print(f"Validation set: {len(eval_df)} examples")
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    eval_dataset = Dataset.from_pandas(eval_df.reset_index(drop=True))
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": eval_dataset
    })
    
    # Load model with frozen phase 1 adapters
    print("\nLoading model with frozen phase 1 adapters...")
    model, tokenizer = load_model_with_frozen_adapters()
    
    # Preprocess data
    def preprocess_batch(examples):
        return preprocess_reasoning_function(examples, tokenizer)
    
    print("\nTokenizing datasets (with truncation)...")
    tokenized_datasets = dataset_dict.map(
        preprocess_batch,
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
        desc="Tokenizing datasets",
    )
    
    print(f"Tokenized training examples: {len(tokenized_datasets['train'])}")
    print(f"Tokenized validation examples: {len(tokenized_datasets['validation'])}")
    
    # Check sequence lengths
    train_lengths = [len(x['input_ids']) for x in tokenized_datasets['train']]
    val_lengths = [len(x['input_ids']) for x in tokenized_datasets['validation']]
    print(f"\nSequence length stats (after truncation):")
    print(f"Train - Min: {min(train_lengths)}, Max: {max(train_lengths)}, Mean: {np.mean(train_lengths):.1f}")
    print(f"Val   - Min: {min(val_lengths)}, Max: {max(val_lengths)}, Mean: {np.mean(val_lengths):.1f}")
    
    # Use custom data collator
    data_collator = CustomDataCollator(tokenizer, max_length=512)
    
    # Training arguments for phase 2 - EXTREME memory optimization
    training_args = TrainingArguments(
        output_dir="./qwen1.5b-clinical-reasoning2.0",
        per_device_train_batch_size=2,  # Reduced from 2
        per_device_eval_batch_size=2,   # Reduced from 2
        gradient_accumulation_steps=4,  # Increased from 4
        learning_rate=1e-5,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=5,
        eval_steps=50,
        save_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=20,
        fp16=True,
        dataloader_pin_memory=False,
        save_total_limit=1,
        remove_unused_columns=False,
        report_to="none",
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        dataloader_drop_last=True,
        gradient_checkpointing=True,  # Enable gradient checkpointing
        eval_accumulation_steps=2,    # Reduce memory during evaluation
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,  # Fixed deprecation warning
    )
    
    # Train phase 2
    print("\n" + "="*80)
    print("STARTING PHASE 2 TRAINING (MEMORY OPTIMIZED)")
    print("="*80)
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Max sequence length: 512 tokens")
    print(f"Gradient checkpointing: Enabled")
    
    try:
        train_result = trainer.train()
        
        # Save final model
        print("\nSaving final model...")
        trainer.save_model()
        tokenizer.save_pretrained("./qwen1.5b-clinical-reasoning")
        
        # Save both adapters
        model.save_pretrained("./qwen1.5b-clinical-reasoning/final_adapters")
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print("Final model saved in: ./qwen1.5b-clinical-reasoning")
        print("Final adapters saved in: ./qwen1.5b-clinical-reasoning/final_adapters")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        print("\nFURTHER MEMORY OPTIMIZATION OPTIONS:")
        print("1. Reduce max_length to 256 in preprocess_reasoning_function")
        print("2. Use smaller model: Qwen/Qwen2.5-0.5B-Instruct")
        print("3. Reduce gradient_accumulation_steps to 4")
        print("4. Use CPU offloading (slower but uses less GPU memory)")
        print("5. Reduce dataset size")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
