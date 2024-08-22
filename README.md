
<p align="left"> 
  <img src="https://github.com/01-ai/Yi/blob/main/assets/img/coder.gif?raw=true" alt="Empowering Code with Precision and Performance" width="800"/> 
</p>


# Introduction to Yi-Coder
Yi Coder is a suite of advanced code language models, meticulously trained from the ground up on a dataset of 3 trillion tokens, with 2 trillion specifically dedicated to code. We offer two model sizes: 1.5B and 9B parameters. Each model is pre-trained on a project-level code corpus utilizing a 128K token context window and an additional fill-in-the-blank task. Yi Coder supports a wide range of functions, including code completion, generation, interpretation, web search, and repository-level code Q&A, catering to various software development scenarios. Yi Coder has demonstrated highly competitive performance on public benchmarks such as LiveCodeBench and HumanEval.

## Highlights：
- **Powerful Coding Performance**: Yi Coder excels in models with less than 10 billion parameters, achieving performance on par with DeepSeek-Coder 33B and Qwen Coder 7B, delivering outstanding code generation capabilities.
- **Support Long Context Window**: Yi Coder supports a long context window of up to 128K tokens, with a custom "needle-in-the-code" feature that consistently delivers excellent, fully green results in testing.

# News
🔥 **2024-09-04**: The Yi-Coder model is open sourced and available to the public.

# Quick Start
## Hardware and software requirements
To set up the environment and install the required packages, execute the following command.
```bash
git clone https://github.com/01-ai/Yi-Coder.git
cd Yi-Coder
pip install -r requirements.txt
```
## transformers
You can use transformers to quickly reason Yi chat or the base model to reason as follows.
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" # the device to load the model onto
model_path = <Huggingface>

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
model_path = <Huggingface>

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
Yi-Coder is now available on Ollama! Please install the latest version of Ollama and run the following command:
```bash
Ollama run Yi-Coder
```

# Cookbook
| Category                    | Description                  | Notebook, Markdown                                                                                                                  | 
|:----------------------------|:-----------------------------|:--------------------------------------------------------------------------------------------------------------------------|
| System prompt              | Perform efficient inference with the Yi model using Swift.    | [Inference_using_swift.ipynb](./en/opensource/Inference/Inference_using_swift.ipynb)                                      | 
| Webpage Artifacts      | Utilize Transformers to rapidly infer the Yi model.  | [Inference_using_transformers.ipynb](./en/opensource/Inference/Inference_using_transformers.ipynb)                           |
| NL2SQL           | Experience fast inference with the Yi model through Imdeploy.  | [Inference_using_lmdeploy.ipynb](./en/opensource/Inference/Inference_using_lmdeploy.ipynb)                                | 
| fine-tune              | Deploy vllm for quick and efficient inference with the Yi model.       | [vLLM_Inference_tutorial.ipynb](./en/opensource/Inference/vLLM_Inference_tutorial.ipynb)                                  | 
| quantization          | Learn how to quantize your Yi model using the power of Swift.         | [swift-yi-quantization.md](./en/opensource/quantization/swift-yi-quantization.md)                                           |
# Training procedure

# License
The code and weights of the Yi-Coder series models are distributed under the Apache 2.0 license.

If you create derivative works based on this model, please include the following attribution in your derivative works:

```This work is a derivative of [The Yi-Coder Series Model You Base On] by 01.AI, used under the Apache 2.0 License.```

# Star
