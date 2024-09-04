from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:63342",
    "http://localhost:8080",
    "*",  # Allow all origins for demonstration purposes. Be cautious in production.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "01-ai/Yi-Coder-9B-Chat"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()


class RequestBody(BaseModel):
    prompt: list


# System instructions
SYSTEM_INSTRUCTIONS = """
You are a helpful AI assistant, please adhere to the following guidelines:
1. When you need to provide HTML code, make sure you provide complete and fully functional HTML code. Include the HTML code in the ``html and ```` tags. Ensure that the generated HTML code is self-contained and can be run independently. 
2. Strictly follow the user instructions and output all code as required. 
3. Your name is the Yi-Coder model. 
4. Include all necessary CSS and JavaScript in the HTML file, making sure not to write them separately.
5. You can also communicate with users normally.
"""

@app.post("/generate")
async def generate_text(request: RequestBody):
    print(request.prompt)
    conversation = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}] + request.prompt
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3096,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id
        )
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"generated_text": generated_text}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
