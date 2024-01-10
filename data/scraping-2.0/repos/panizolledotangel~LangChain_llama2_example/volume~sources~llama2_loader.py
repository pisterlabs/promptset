# Codigo de la web https://www.pinecone.io/learn/llama-2/

import os
import torch
import transformers
from langchain.llms import HuggingFacePipeline

CACHE_DIR = "/home/angel/host_data/.cache/huggingface/hub"

def load_llama_llm(model_id: str = 'meta-llama/Llama-2-70b-chat-hf') -> HuggingFacePipeline:

    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # begin initializing HF items, need auth token for these
    hf_auth = os.environ["HUGGING_TOKEN"]
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        cache_dir=CACHE_DIR,
        use_auth_token=hf_auth
    )

    # initialize the model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=hf_auth
    )
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=hf_auth,
        cache_dir=CACHE_DIR
    )

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=4000,  # mex number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )

    llm = HuggingFacePipeline(pipeline=generate_text)
    return llm
