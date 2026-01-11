import gradio as gr
import spaces
from unsloth import FastLanguageModel
import torch

# Global model/tokenizer (loaded once on first GPU call)
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None:
        model_id = "NurseCitizenDeveloper/NurseSim-Triage-Llama-3.2-3B"
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    return model, tokenizer

# ZeroGPU decorator - GPU is only used when this function is called
@spaces.GPU(duration=60)
def triage_patient(complaint, hr, bp, spo2, temp):
    model, tokenizer = load_model()
    
    prompt = f"""### Instruction:
You are an expert A&E Triage Nurse. Assess the following patient and provide your triage decision.

### Input:
Patient Complaint: {complaint}
Vitals: HR {hr}, BP {bp}, SpO2 {spo2}%, Temp {temp}C.

### Response:"""

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256,
        use_cache=True,
    )
    
    response = tokenizer.batch_decode(outputs)[0]
    
    if "### Response:" in response:
        response = response.split("### Response:")[1].strip()
    
    return response

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🩺 NurseSim AI: Emergency Triage Simulator
    **An AI agent fine-tuned for the Manchester Triage System (MTS).**
    *Developed for the OpenEnv Challenge by NurseCitizenDeveloper.*
    
    > ⚡ Powered by **ZeroGPU** - Model loads on-demand for efficient inference.
    """)
    
    with gr.Row():
        with gr.Column():
            complaint = gr.Textbox(label="Chief Complaint", placeholder="e.g., Shortness of breath and chest tightness...")
            with gr.Row():
                hr = gr.Number(label="Heart Rate", value=80)
                bp = gr.Textbox(label="Blood Pressure", placeholder="e.g., 120/80")
            with gr.Row():
                spo2 = gr.Slider(label="SpO2 (%)", minimum=50, maximum=100, value=98)
                temp = gr.Number(label="Temperature (C)", value=37.0)
            
            submit_btn = gr.Button("Assess Patient", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(label="AI Triage Assessment", lines=10)
            gr.Markdown("""
            ### ⚠️ Safety Warning
            This is a research prototype for the **OpenEnv Challenge**. 
            It is based on a fine-tuned Llama 3.2 model and is **NOT** a certified medical device.
            """)

    submit_btn.click(
        fn=triage_patient,
        inputs=[complaint, hr, bp, spo2, temp],
        outputs=output_text
    )
    
    gr.Examples(
        examples=[
            ["Crushing chest pain and nausea", 110, "90/60", 94, 37.2],
            ["Twisted ankle at football, walking with difficulty", 75, "125/85", 99, 36.8],
            ["High fever and confusion in elderly patient", 105, "100/70", 92, 39.5],
        ],
        inputs=[complaint, hr, bp, spo2, temp]
    )

if __name__ == "__main__":
    demo.launch()
