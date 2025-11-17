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

def load_and_combine_symptoms_data():
    """Load and combine symptoms data from CSV files"""
    try:
        # Load symptoms data
        symptoms_df = pd.read_csv("disease-symptom/DiseaseAndSymptoms.csv")
        print(f"Symptoms columns: {list(symptoms_df.columns)}")
        
        # Load precautions data
        precautions_df = pd.read_csv("disease-symptom/Disease precaution.csv")
        print(f"Precautions columns: {list(precautions_df.columns)}")
        
        # Combine all symptom columns (Symptom_1 to Symptom_17) into one comma-separated column
        symptom_columns = [f'Symptom_{i}' for i in range(1, 18)]
        available_symptom_columns = [col for col in symptom_columns if col in symptoms_df.columns]
        print(f"Available symptom columns: {available_symptom_columns}")
        
        symptoms_df['combined_symptoms'] = symptoms_df[available_symptom_columns].apply(
            lambda x: ', '.join([str(symptom) for symptom in x if pd.notna(symptom) and str(symptom) != 'nan' and str(symptom) != '']), 
            axis=1
        )
        
        # Combine precautions columns
        precaution_columns = [f'Precaution_{i}' for i in range(1, 5)]
        available_precaution_columns = [col for col in precaution_columns if col in precautions_df.columns]
        print(f"Available precaution columns: {available_precaution_columns}")
        
        precautions_df['combined_precautions'] = precautions_df[available_precaution_columns].apply(
            lambda x: ', '.join([str(precaution) for precaution in x if pd.notna(precaution) and str(precaution) != 'nan' and str(precaution) != '']), 
            axis=1
        )
        
        # Merge the datasets
        merged_df = pd.merge(
            symptoms_df[['Disease', 'combined_symptoms']],
            precautions_df[['Disease', 'combined_precautions']],
            on='Disease',
            how='inner'
        )
        
        # Remove empty symptom/precaution entries
        merged_df = merged_df[
            (merged_df['combined_symptoms'].str.strip() != '') & 
            (merged_df['combined_precautions'].str.strip() != '')
        ]
        
        return merged_df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        raise

def load_model_and_tokenizer():
    """Load Qwen model with 4-bit quantization"""
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    )
    
    print("Loading model in 4-bit quantization mode...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def setup_lora_phase1(model):
    """Configure LoRA for first phase training"""
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True
    )
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

class CustomDataCollator:
    """Custom data collator that handles padding properly for variable length sequences"""
    def __init__(self, tokenizer, max_length=512):
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

def preprocess_symptoms_function(examples, tokenizer):
    """Preprocess symptoms and precautions data for training"""
    texts = []
    
    for i in range(len(examples['Disease'])):
        instruction = "Given the disease, identify symptoms and precautions:"
        disease = examples['Disease'][i]
        symptoms = examples['combined_symptoms'][i]
        precautions = examples['combined_precautions'][i]
        
        full_text = f"<|im_start|>user\n{instruction}\n\nDisease: {disease}<|im_end|>\n<|im_start|>assistant\nSymptoms: {symptoms}\nPrecautions: {precautions}<|im_end|>"
        texts.append(full_text)
    
    # Tokenize without padding first
    tokenized = tokenizer(
        texts,
        truncation=False,  # We'll handle truncation in collator
        padding=False,
        max_length=None,
        return_tensors=None,
        add_special_tokens=True,
    )
    
    # Create labels - for causal LM, labels are the same as input_ids
    # We'll mask the user part during training via the loss function
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
    print("PHASE 1: SYMPTOMS & PRECAUTIONS TRAINING")
    print("="*80)
    
    # Load and prepare data
    print("\nLoading symptoms and precautions data...")
    df = load_and_combine_symptoms_data()
    
    print(f"\nTotal examples: {len(df)}")
    print(f"Sample data:")
    print(df.head(2))
    
    # Check for any issues
    print(f"\nData quality check:")
    print(f"Empty symptoms: {len(df[df['combined_symptoms'].str.strip() == ''])}")
    print(f"Empty precautions: {len(df[df['combined_precautions'].str.strip() == ''])}")
    print(f"Unique diseases: {df['Disease'].nunique()}")
    
    # Split data
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"\nTraining set: {len(train_df)} examples")
    print(f"Validation set: {len(eval_df)} examples")
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    eval_dataset = Dataset.from_pandas(eval_df.reset_index(drop=True))
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": eval_dataset
    })
    
    # Load model and tokenizer
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer()
    
    # Setup LoRA for phase 1
    print("\nSetting up LoRA for phase 1...")
    model = setup_lora_phase1(model)
    
    # Preprocess data
    def preprocess_batch(examples):
        return preprocess_symptoms_function(examples, tokenizer)
    
    print("\nTokenizing datasets...")
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
    print(f"\nSequence length stats:")
    print(f"Train - Min: {min(train_lengths)}, Max: {max(train_lengths)}, Mean: {np.mean(train_lengths):.1f}")
    print(f"Val   - Min: {min(val_lengths)}, Max: {max(val_lengths)}, Mean: {np.mean(val_lengths):.1f}")
    
    # Use custom data collator
    data_collator = CustomDataCollator(tokenizer, max_length=512)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./qwen1.5b-symptoms-precautions",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=2,
        logging_dir="./logs",
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=30,
        fp16=True,
        dataloader_pin_memory=False,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        dataloader_drop_last=True,  # Prevent last incomplete batch
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train phase 1
    print("\n" + "="*80)
    print("STARTING PHASE 1 TRAINING")
    print("="*80)
    
    try:
        train_result = trainer.train()
        
        # Save phase 1 model
        print("\nSaving phase 1 model...")
        trainer.save_model()
        tokenizer.save_pretrained("./qwen1.5b-symptoms-precautions")
        
        # Save LoRA adapters for phase 1
        model.save_pretrained("./qwen1.5b-symptoms-precautions/phase1_adapters")
        
        print(f"\n{'='*80}")
        print("PHASE 1 TRAINING COMPLETED!")
        print(f"{'='*80}")
        print("Model saved in: ./qwen1.5b-symptoms-precautions")
        print("Adapters saved in: ./qwen1.5b-symptoms-precautions/phase1_adapters")
        print(f"Total training examples: {len(train_df)}")
        print(f"Total validation examples: {len(eval_df)}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        print("Troubleshooting steps:")
        print("1. Check if all sequences have reasonable lengths")
        print("2. Try reducing batch size further")
        print("3. Check GPU memory")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()