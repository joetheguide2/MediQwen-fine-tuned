MediQwen: A Clinical Diagnostic Reasoning Model 

This project implements a two-phase fine-tuned Qwen2.5-1.5B model for medical diagnostic reasoning with enhanced interpretability and reduced hallucination. The model architecture uses Qwen2.5-1.5B-Instruct as the base model with a two-phase LoRA fine-tuning approach.

Phase 1 training utilizes the Disease and Symptoms dataset from Kaggle for disease symptoms and precautions knowledge acquisition. The dataset is available at: https://www.kaggle.com/datasets/choongqianzheng/disease-and-symptoms-dataset

Phase 2 training employs the MIMIC-IV-Ext-DiReCT dataset for clinical reasoning chain learning. This dataset is cited as: 
Wang, Bowen, Chang, Jiuyang, and Yiming Qian. "MIMIC-IV-Ext-DiReCT" (version 1.0.0). PhysioNet (2025). RRID:SCR_007345. https://doi.org/10.13026/yf96-kc87

The technical implementation involves Phase 1 training on disease symptoms and precautions using structured CSV files with LoRA configuration of rank 8 targeting q_proj, k_proj, v_proj, and o_proj modules, resulting in approximately 9.8 million trainable parameters. Phase 2 training focuses on diagnostic reasoning with structured output format using the MIMIC-IV-Ext-DiReCT clinical reasoning chains with LoRA configuration of rank 4 targeting gate_proj and up_proj modules, resulting in approximately 24.5 thousand trainable parameters while keeping Phase 1 adapters frozen.

Training specifications include 4-bit NF4 quantization with double quantization, batch size of 1 per device with gradient accumulation, sequence length of 512 tokens, Paged AdamW 8-bit optimizer, and enabled gradient checkpointing.

The model produces structured diagnostic output with four key components: most suspected disease, other significant risks, diagnostic reasoning chain, and disease-specific precautions. Key features include reduced hallucination through two-phase training that prevents knowledge corruption, interpretable reasoning with explicit reasoning chains for transparency, accurate recall that maintains disease-symptom knowledge from Phase 1, and structured output with consistent formatting for clinical utility.

Performance metrics demonstrate improved reasoning accuracy with better diagnostic chain coherence, maintained knowledge retention of symptom-disease associations, reduced hallucination rate through phased training approach, and enhanced interpretability with explicit reasoning steps for clinical validation.

The file structure includes phase1_adapters for disease knowledge (frozen), phase2_adapters for reasoning capabilities, and final_adapters containing combined adapters.

Results show the two-phase training approach achieves enhanced diagnostic accuracy through explicit reasoning chains, reduced model hallucination via knowledge preservation, improved clinical interpretability with structured reasoning output, and maintained symptom-disease recall from foundational training.

Limitations include training data constrained to available clinical reasoning chains, performance dependent on base model and first dataset clinical knowledge, and requirement for validation with clinical experts for real-world deployment.
