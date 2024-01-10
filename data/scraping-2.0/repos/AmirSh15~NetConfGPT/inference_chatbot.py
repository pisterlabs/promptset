import os
import time
from typing import List
import warnings
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
    AutoConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatModel

from utils.utils import remove_descriptions

warnings.filterwarnings("ignore", category=UserWarning)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_OUTPUT_LENGTH = 2000
TOP_K = 50
TOP_P = 0.95
TEMPERATURE = 0.1
SAVE_PATH = "/home/amir/NetConfGPT/results/"
access_token = "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"
EXTENDER = "_ZERO_SHOT"

class StopGenerationCriteria(StoppingCriteria):
    def __init__(
        self, tokens: List[List[str]], tokenizer: AutoTokenizer, device: torch.device
    ):
        stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
        self.stop_token_ids = [
            torch.tensor(x, dtype=torch.long, device=device) for x in stop_token_ids
        ]
 
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False

# first purge cuda memory
torch.cuda.empty_cache()

# check if SAVE_PATH exists
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH, exist_ok=True)

# Prepare quantized config
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the model
model_id = "mistralai/Mistral-7B-Instruct-v0.1" # "mistralai/Mistral-7B-Instruct-v0.1" # "mistralai/Mistral-7B-v0.1" # "bigcode/starcoderbase-1b" # "Salesforce/codegen25-7b-multi" # "cerebras/btlm-3b-8k-base" # "meta-llama/Llama-2-70b-chat-hf" # "tiiuae/falcon-40b"
model_name = model_id.split("/")[-1]
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # load_in_8bit_fp32_cpu_offload=True, # for falcon
    # load_in_4bit=True, # for QLoRA
    quantization_config=nf4_config,
    trust_remote_code=True,
    device_map="auto",
    token=access_token,
)
model = model.eval()

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# set generation configs
generation_config = model.generation_config
generation_config.temperature = 0
generation_config.num_return_sequences = 1
generation_config.max_new_tokens = 256
generation_config.use_cache = False
generation_config.repetition_penalty = 1.7
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

config = AutoConfig.from_pretrained(model_id)
model_max_length = config.max_position_embeddings
logger.info(f"max_length: {model_max_length}")

prompt = """
The following is a friendly conversation between a human and an AI. The AI is
talkative and provides lots of specific details from its context.
 
Current conversation:
 
Human: Who is Dwight K Schrute?
AI:
""".strip()
 
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.to(model.device)
 
with torch.inference_mode():
    outputs = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
    )
    
# decode the output
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

logger.info(f"manual response: {response}")

stop_tokens = [["Human", ":"], ["AI", ":"]]
stopping_criteria = StoppingCriteriaList(
    [StopGenerationCriteria(stop_tokens, tokenizer, model.device)]
)

generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    task="text-generation",
    stopping_criteria=stopping_criteria,
    generation_config=generation_config,
)
 
llm = HuggingFacePipeline(pipeline=generation_pipeline)

res = llm(prompt)

logger.info(f"pipeline response with stopping criteria: {res}")