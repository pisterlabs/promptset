from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import xformers


# This was inserted by me
import torch
# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')

from constants import CHROMA_SETTINGS

from torch import cuda, bfloat16
import transformers

def run_model(query):
    #   'meta-llama/Llama-2-70b-chat-hf'    bigscience/bloom-560m
    model_id = 'lmsys/vicuna-7b-v1.3'
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    # Set quantization configuration to load large model with less GPU memory  - Cannot use quantization in Windows
    # bnb_config = transformers.BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type='nf4',
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=bfloat16
    # )

    # Initialize model configuration and model
    hf_token = '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'
    model_config = transformers.AutoConfig.from_pretrained(model_id, use_auth_token=hf_token)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        # quantization_config=bnb_config,
        device_map='auto'
    )
    model.eval()

    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    # Define text generation pipeline
    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,  # Langchain expects the full text
        task='text-generation',
        do_sample=True,
        temperature=0.1,  # 'Randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # Max number of tokens to generate in the output
        repetition_penalty=1.1  # Without this, output begins repeating
    )

    # Generate text based on the input query
    res = generate_text(query)
    answer = res[0]["generated_text"]

    return answer
