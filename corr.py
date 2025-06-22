from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os,pylab,torch,scipy

llms = {
    # "google/gemma-3-4b-it": [34,8,256,2],
    # "google/gemma-3-1b-it": [26,4,256,4],
    # "google/gemma-2-2b": [26,8,256,2],
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":[28,12,128,6],
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":[32,32,128,4],
    # "google/gemma-2-2b-it": [26,8,256,2],
    # "meta-llama/Llama-3.2-1B-Instruct": [16,32,64,4],    
    # "meta-llama/Llama-3.2-3B-Instruct": [28,24,128,3],
    # "Qwen/Qwen2.5-0.5B-Instruct": [24,14,64,7],
    # "Qwen/Qwen2.5-1.5B-Instruct": [28,12,128,6],
    # "mistralai/Mistral-7B-Instruct-v0.3": [32,32,128,4],
    # "google/gemma-2-9b-it": [42,16,256,2],
    # "google/gemma-2b-it": [18,8,256,8],    
    # "google/gemma-7b-it": [28,16,256,1],
    # "Qwen/Qwen2.5-7B-Instruct": [28,28,128,7],
    # "google/gemma-2-27b": [46,32,128,2],
    # "meta-llama/Llama-3.3-70B-Instruct": [80,64,128,8],
    # "meta-llama/Meta-Llama-3.1-8B-Instruct": [32,32,128,4],
    # "meta-llama/Meta-Llama-3.1-8B": [32,32,128,4],
    # "meta-llama/Meta-Llama-3-8B-Instruct": [32,32,128,4],
    # "meta-llama/Meta-Llama-3-8B": [32,32,128,4],
    # "google/gemma-2b": [18,8,256,8],
    # "google/gemma-7b": [28,16,256,1],
    # "google/gemma-2-9b": [42,16,256,2],
    # "mistralai/Mistral-7B-v0.3": [32,32,128,4],
    # "meta-llama/Llama-3.2-1B": [16,32,64,4],
    # "meta-llama/Llama-3.2-3B": [28,24,128,3],
    # "Qwen/Qwen2.5-0.5B": [24,14,64,7],
    # "Qwen/Qwen2.5-1.5B": [28,12,128,6],
    # "Qwen/Qwen2.5-7B": [28,28,128,7],
}

for llm in llms.keys():
    model = AutoModelForCausalLM.from_pretrained(llm,torch_dtype=torch.bfloat16,device_map="auto")
    nb,nh,hdim,hratio = llms[llm]
    print_fig(nb=nb,nh=nh,hdim=hdim,hratio=hratio,model=model)
    pylab.suptitle(llm)
    pylab.savefig(llm.split('/')[-1]+'.png')
    print(llm)
