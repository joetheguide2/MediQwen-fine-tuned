"""
Production FastAPI Server for Clinical Reasoning Model
Deploy this on a server and access from Android/Web

Installation:
    pip install fastapi uvicorn transformers torch peft bitsandbytes python-multipart

Run locally:
    python api_server.py

Run in production:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 2
"""

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    MODEL_PATH = "./qwen1.5b-clinical-reasoning"
    ADAPTER_PATH = "./qwen1.5b-clinical-reasoning/adapters"
    BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # API Settings
    API_KEY = "your-secret-api-key-change-this"  # Change in production!
    MAX_TOKENS = 200
    DEFAULT_TEMPERATURE = 0.7
    
    # Server Settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_4BIT = True  # Use 4-bit quantization to save memory

config = Config()

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Clinical Reasoning API",
    description="AI-powered clinical diagnostic reasoning assistant",
    version="1.0.0"
)

# Add CORS middleware for web/mobile access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ClinicalNoteRequest(BaseModel):
    clinical_note: str = Field(..., min_length=10, description="Clinical note to analyze")
    max_tokens: Optional[int] = Field(200, ge=50, le=500, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, ge=0.1, le=2.0, description="Temperature for generation")
    
    class Config:
        schema_extra = {
            "example": {
                "clinical_note": "Chest pain\n\nPatient is a 65 year old male with history of HTN, diabetes who presents with chest pain. Pain started 2 hours ago while at rest, 7/10 severity.\n\nLabs: Troponin-T 0.55",
                "max_tokens": 200,
                "temperature": 0.7
            }
        }

class ClinicalNoteResponse(BaseModel):
    reasoning: str
    success: bool
    inference_time: float
    timestamp: str
    model_version: str = "qwen-clinical-1.0"
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str

# ============================================================================
# MODEL LOADING
# ============================================================================

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.mobile_config = None
        self.is_loaded = False
    
    def load_model(self):
        """Load the fine-tuned model - exactly like inference_terminal.py"""
        if self.is_loaded:
            logger.info("Model already loaded")
            return
        
        logger.info("Loading Clinical Reasoning Model...")
        start_time = time.time()
        
        try:
            # Load tokenizer first
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.MODEL_PATH,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            logger.info(f"Tokenizer vocabulary size: {len(self.tokenizer)}")
            
            # Configure quantization
            if config.USE_4BIT and config.DEVICE == "cuda":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
                logger.info("Using 4-bit quantization")
            else:
                bnb_config = None
                logger.info("Using full precision")
            
            # Load base model
            logger.info("Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                config.BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto" if config.DEVICE == "cuda" else "cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Resize embeddings
            base_model.resize_token_embeddings(len(self.tokenizer))
            logger.info("Resized model embeddings")
            
            # Load LoRA adapters
            logger.info(f"Loading LoRA adapters from {config.ADAPTER_PATH}...")
            self.model = PeftModel.from_pretrained(base_model, config.ADAPTER_PATH)
            self.model.eval()
            
            # Load mobile config if exists
            import json
            import os
            mobile_config_path = f"{config.MODEL_PATH}/mobile_config.json"
            if os.path.exists(mobile_config_path):
                with open(mobile_config_path, 'r') as f:
                    self.mobile_config = json.load(f)
                logger.info(f"Mobile config loaded: {self.mobile_config['mode']}")
            
            load_time = time.time() - start_time
            self.is_loaded = True
            
            logger.info(f"✅ Model loaded successfully in {load_time:.2f}s")
            logger.info(f"Device: {config.DEVICE}")
            
            if config.DEVICE == "cuda":
                memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                logger.info(f"GPU Memory: {memory_allocated:.2f} GB")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_reasoning(self, clinical_note: str, max_tokens: int, temperature: float) -> str:
        """Generate diagnostic reasoning - exactly like inference_terminal.py"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Adjust max_tokens based on mobile config
        if self.mobile_config:
            if self.mobile_config['mode'] == 'extreme':
                max_tokens = min(max_tokens, 100)
            elif self.mobile_config['mode'] == 'ultra_light':
                max_tokens = min(max_tokens, 150)
        
        # Format input - EXACT same as inference_terminal.py
        instruction = "Analyze this clinical note and provide diagnostic reasoning:"
        prompt = f"<|im_start|>user\n{instruction}\n\nClinical Note:\n{clinical_note}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        max_input_length = self.mobile_config['max_seq_length'] if self.mobile_config else 1024
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=max_input_length, truncation=True)
        
        # Move to device
        if config.DEVICE == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
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

# Initialize model manager
model_manager = ModelManager()

# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model when server starts"""
    logger.info("="*80)
    logger.info("Starting Clinical Reasoning API Server")
    logger.info("="*80)
    try:
        model_manager.load_model()
        logger.info("Server ready to accept requests")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

# ============================================================================
# API ENDPOINTS
# ============================================================================

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key for authentication"""
    if config.API_KEY != "your-secret-api-key-change-this":  # Only check if changed from default
        if x_api_key != config.API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Clinical Reasoning API",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze": "Analyze clinical note",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "initializing",
        model_loaded=model_manager.is_loaded,
        device=config.DEVICE,
        timestamp=datetime.now().isoformat()
    )

@app.post("/analyze", response_model=ClinicalNoteResponse)
async def analyze_clinical_note(
    request: ClinicalNoteRequest,
    x_api_key: str = Header(None)
):
    """
    Analyze a clinical note and generate diagnostic reasoning
    
    **Authentication:** Include `X-API-Key` header with your API key
    """
    # Verify API key
    verify_api_key(x_api_key)
    
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet, please try again")
    
    try:
        start_time = time.time()
        
        logger.info(f"Analyzing clinical note (length: {len(request.clinical_note)} chars)")
        
        # Generate reasoning
        reasoning = model_manager.generate_reasoning(
            request.clinical_note,
            request.max_tokens,
            request.temperature
        )
        
        inference_time = time.time() - start_time
        
        logger.info(f"Analysis complete in {inference_time:.2f}s")
        
        return ClinicalNoteResponse(
            reasoning=reasoning,
            success=True,
            inference_time=round(inference_time, 2),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return ClinicalNoteResponse(
            reasoning="",
            success=False,
            inference_time=0.0,
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )

@app.post("/batch-analyze")
async def batch_analyze(
    notes: list[ClinicalNoteRequest],
    x_api_key: str = Header(None)
):
    """Analyze multiple clinical notes in batch"""
    verify_api_key(x_api_key)
    
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for note in notes:
        try:
            start_time = time.time()
            reasoning = model_manager.generate_reasoning(
                note.clinical_note,
                note.max_tokens or 200,
                note.temperature or 0.7
            )
            inference_time = time.time() - start_time
            
            results.append({
                "reasoning": reasoning,
                "success": True,
                "inference_time": round(inference_time, 2)
            })
        except Exception as e:
            results.append({
                "reasoning": "",
                "success": False,
                "error": str(e)
            })
    
    return {"results": results, "count": len(results)}

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║                   CLINICAL REASONING API SERVER                                ║
╚════════════════════════════════════════════════════════════════════════════════╝

Starting server...

Once running, access:
  - API Documentation: http://localhost:8000/docs
  - Health Check: http://localhost:8000/health
  - Root: http://localhost:8000

API Key: {config.API_KEY}
(Change this in production!)

Device: {config.DEVICE}
    """)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )