
<p align="center"> 
  <img src="https://github.com/01-ai/Yi/blob/main/assets/img/yicoder.gif?raw=true" alt="yicoder" width="500"/> 
</p>


# Introduction to Yi-Coder
**Yi-Coder series models** are trained for coding tasks with two sizes available, 1.5B and 9B, supporting 52 major coding languages. Notably, the Yi-Coder-9B outperforms other models under 10 billion parameters such as CodeQwen1.5 7B and CodeGeex4 9B, and even achieves performance on par with DeepSeek-Coder 33B.

Yi-Coder excels in long-context understanding, handling up to 128K tokens for project-level code comprehension and generation. Despite its relatively small size, Yi-coder is versatile in tasks like programming, code editing, debugging, completion, and mathematical reasoning. 

<p align="left">
  <img src="https://github.com/01-ai/Yi-Coder/blob/main/assets/demo1.gif?raw=true" alt="demo1" width="400"/>
  <img src="https://github.com/01-ai/Yi-Coder/blob/main/assets/demo2.gif?raw=true" alt="demo2" width="400"/>
</p>

For more details, see Yi-Coder blog.

# News
🔥 **2024-09-05**: The Yi-Coder series models are open sourced and available to the public.

# Quick Start
## Requirements
Make sure you have `python>=3.9` installed before using it.To set up the environment and install the requirements, run the following command: 
```bash
git clone https://github.com/01-ai/Yi-Coder.git
cd Yi-Coder
pip install -r requirements.txt
```
## Transformers
You can use transformers to run inference with Yi-Coder models (both chat and base versions) as follows:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" # the device to load the model onto
model_path = "01-ai/Yi-Coder-9B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto").eval()

prompt = "Write a quick sort algorithm."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=1024,
    eos_token_id=tokenizer.eos_token_id  
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```
## vllm
You can also use vLLM to reason about Yi-Coder models. vLLM is a fast and easy-to-use library for reasoning about and serving large language models (LLMs). Be sure to install vLLM and then do the following
```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
model_path = "01-ai/Yi-Coder-9B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_path)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.8)

llm = LLM(model=model_path, 
          gpu_memory_utilization=0.9, 
          max_model_len=1024)

prompt = "Write a quick sort algorithm."  
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(text)

# Generate the response
outputs = llm.generate([text], sampling_params)

# Print the output
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

```
## ollama
Yi-Coder series models are now available on Ollama! Install the latest version of Ollama and run the following command: 
```bash
Ollama run Yi-Coder
```

# Cookbook
- [System prompt](https://github.com/01-ai/Yi-Coder/blob/main/cookbook/System_prompt/System_prompt.ipynb): Enhance coding workflow with code completion, insertion, and quality assurance.
- [Webpage](https://github.com/01-ai/Yi-Coder/blob/main/cookbook/Webpage/Webpage.md): Turn your ideas into web pages!
- [NL2SQL](https://github.com/01-ai/Yi-Coder/blob/main/cookbook/NL2SQL/NL2SQL.md): Convert natural language queries into Structured Query Language (SQL).
- [Fine-tune](https://github.com/01-ai/Yi-Coder/blob/main/cookbook/Fine_tune/finetune.md): Fine-tune the Yi-Coder series models for your specific needs.
- [Quantization](https://github.com/01-ai/Yi-Coder/blob/main/cookbook/Quantization/quantization.md): Quantize your Yi-Coder series models using Swift.


# License
The code and weights of the Yi-Coder series models are distributed under the Apache 2.0 license.

If you create derivative works based on this model, please include the following attribution in your derivative works:

```This work is a derivative of [The Yi Series Model You Based On] by 01.AI, licensed under the Apache 2.0 License.```

# Star
 🚀 Propser with Yi-Coder—star it! 🌟
