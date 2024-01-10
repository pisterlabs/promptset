import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from langchain import HuggingFacePipeline

from accelerate import infer_auto_device_map, init_empty_weights
from huggingface_hub import login

from transformers import pipeline


def get_prompt(instruction, prompt):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    prompt_template =  B_INST + B_SYS + instruction + E_SYS + prompt + E_INST
    return prompt_template


def load_model(model_id = 'nick-1234/llama-2-7b-finetuned-for-news_comments_generation', huggingface_token = None):
    if (huggingface_token != None):
      login(huggingface_token)
    config = AutoConfig.from_pretrained(model_id)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    #CPU : float32
    #GPU float16
    device_map = infer_auto_device_map(model,dtype="float16",verbose=True)
    #CPU : float32
    #GPU float16
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                device_map=device_map,
                                                offload_folder="offload",
                                                cache_dir = "cache/",
                                                torch_dtype=torch.float16,
                                                offload_state_dict=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                              add_eos_token=True,
                                              cache_dir = "cache/"
                                              )
    return model, tokenizer

def send_prompt_llama2(instructions, prompt, model, tokenizer, temp = 0.5, max_tokens = 100):
  pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens = max_tokens,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )
  final_prompt = get_prompt(instructions, prompt)
  llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':temp})
  output = llm(final_prompt)
  return output