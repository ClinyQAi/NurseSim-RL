#!/usr/bin/env python3
"""
NurseSim-Triage A2A Agent Entry Point

This module implements the Agent-to-Agent (A2A) protocol interface for NurseSim-Triage.
It creates a persistent FastAPI server to handle requests from the AgentBeats platform.
"""

import os
import json
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# Data Models
# ==========================================

class Vitals(BaseModel):
    heart_rate: int = 80
    blood_pressure: str = "120/80"
    spo2: int = 98
    temperature: float = 37.0

class TaskInput(BaseModel):
    complaint: str
    vitals: Vitals

# ==========================================
# Agent Core Logic
# ==========================================

class NurseSimTriageAgent:
    """
    A2A-compatible triage agent for clinical assessment.
    """
    
    def __init__(self):
        """Initialize the triage agent and load the model."""
        self.model = None
        self.tokenizer = None
        self.HF_TOKEN = os.environ.get("HF_TOKEN")
        
        if not self.HF_TOKEN:
            print("WARNING: HF_TOKEN not set. Model loading will fail if authentication is required.")
        
        self._load_model()
    
    def _load_model(self):
        """Load the base model and LoRA adapters."""
        if self.model is not None:
            return  # Already loaded
        
        try:
            base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
            adapter_id = "NurseCitizenDeveloper/NurseSim-Triage-Llama-3.2-3B"
            
            print(f"Loading tokenizer from {adapter_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                adapter_id,
                token=self.HF_TOKEN
            )
            
            print(f"Loading base model {base_model_id}...")
            # Use device_map="auto" to handle CPU/GPU automatically
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True,
                token=self.HF_TOKEN,
            )
            
            print(f"Applying LoRA adapters from {adapter_id}...")
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_id,
                token=self.HF_TOKEN
            )
            self.model.eval()
            print(f"Model loaded successfully on {self.model.device}!")
        except Exception as e:
            print(f"CRITICAL ERROR loading model: {e}")
            # We don't raise here to allow the server to start (and report unhealthy status)
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process an A2A task and return the triage assessment."""
        if self.model is None:
            return {"error": "Model not loaded", "triage_category": "Error"}

        try:
            # Extract task data
            complaint = task.get("complaint", "")
            vitals = task.get("vitals", {})
            hr = vitals.get("heart_rate", 80)
            bp = vitals.get("blood_pressure", "120/80")
            spo2 = vitals.get("spo2", 98)
            temp = vitals.get("temperature", 37.0)
            
            # Create prompt
            prompt = f"""### Instruction:
You are an expert A&E Triage Nurse. Assess the following patient and provide your triage decision.

### Input:
Patient Complaint: {complaint}
Vitals: HR {hr}, BP {bp}, SpO2 {spo2}%, Temp {temp}C.

### Response:"""
            
            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()
            
            triage_category = self._extract_triage_category(response)
            
            return {
                "triage_category": triage_category,
                "assessment": response,
                "recommended_action": self._extract_recommended_action(response)
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "triage_category": "Error",
                "assessment": f"Error processing task: {str(e)}"
            }
    
    def _extract_triage_category(self, response: str) -> str:
        """Extract the triage category from the model's response."""
        response_lower = response.lower()
        if "immediate" in response_lower or "resuscitation" in response_lower: return "Immediate"
        elif "very urgent" in response_lower or "emergency" in response_lower: return "Very Urgent"
        elif "urgent" in response_lower: return "Urgent"
        elif "standard" in response_lower: return "Standard"
        elif "non-urgent" in response_lower or "non urgent" in response_lower: return "Non-Urgent"
        else: return "Standard"
    
    def _extract_recommended_action(self, response: str) -> str:
        if "monitor" in response.lower(): return "Monitor patient closely"
        elif "immediate" in response.lower() or "urgent" in response.lower(): return "Immediate medical attention required"
        else: return "Continue assessment and treatment as per protocol"
    
    def health_check(self) -> Dict[str, Any]:
        """Return agent health status."""
        return {
            "status": "healthy" if self.model is not None else "unhealthy",
            "model_loaded": self.model is not None,
            "gpu_available": torch.cuda.is_available(),
            "device": str(self.model.device) if self.model else "not loaded"
        }

# ==========================================
# FastAPI Server Setup
# ==========================================

print("Initializing NurseSim-Triage Agent...")
agent = NurseSimTriageAgent()

app = FastAPI(
    title="NurseSim-Triage Agent",
    description="A2A Interface for Clinical Triage",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "NurseSim-Triage Agent is running. Visit /health for status."}

@app.get("/health")
async def health_check():
    return agent.health_check()

@app.get("/.well-known/agent-card.json")
async def get_agent_card():
    card_path = ".well-known/agent-card.json"
    if os.path.exists(card_path):
        with open(card_path, "r") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Agent card not found")

@app.post("/process-task")
async def process_task(task: TaskInput):
    """
    Standard A2A task processing endpoint.
    Accepts JSON body matching TaskInput schema.
    """
    result = agent.process_task(task.dict())
    return result

# ==========================================
# Entry Point
# ==========================================

if __name__ == "__main__":
    print("Starting A2A Server on port 8080...")
    # Listen on all interfaces (0.0.0.0) for Docker support
    uvicorn.run(app, host="0.0.0.0", port=8080)
