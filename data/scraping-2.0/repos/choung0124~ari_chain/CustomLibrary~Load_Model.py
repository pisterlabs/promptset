from langchain.llms import RWKV, CTransformers
from transformers import pipeline, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM
from langchain.chains import LLMChain
import gc

def create_gptq_pipeline(model_name_or_path, model_basename, tokenizer):
    # Check if the model exists in memory and delete it
    if 'model' in globals():
        del model
        gc.collect()
        print("The existing model has been deleted from memory.")

    # Load the new model
    model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
            model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            device="cuda:0",
            use_triton=True,
            quantize_config=None)

    return pipeline(
       "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=8192,
        temperature=0.2,
        top_p=0.95,
        repetition_penalty=1.15
        )

def create_8bit_pipeline(model_name_or_path, tokenizer):
    # Check if the model exists in memory and delete it
    if 'model' in globals():
        del model
        gc.collect()
        print("The existing model has been deleted from memory.")

    # Load the new model
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
            trust_remote_code=True,
            device_map="auto",
            load_in_8bit=True)
    
    return pipeline(
       "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2048,
        temperature=0.2,
        top_p=0.95,
        repetition_penalty=1.15
        )

def create_ggml_model(model_name_or_path, model_filename):
    config ={"temperature": 0.2,
             "repetition_penalty": 1.15,
             "max_new_tokens": 2048,
             "context_length": 8192,
             "gpu_layers": 10000000}
    
    llm = CTransformers(model=model_name_or_path, 
                            model_file=model_filename,
                            config=config)
    
    return llm