from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import yaml
import openai
import os
import torch
import replicate
import gradio as gr
from PIL import Image
from io import BytesIO
from transformers import pipeline
import requests
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# def load_config(cfg_file="config.yaml"):
#     with open(cfg_file, "r") as ymlfile:
#         cfg = yaml.safe_load(ymlfile)
#     os.environ['REPLICATE_API_TOKEN'] = cfg["api_keys"]["replicate_token"]
#     os.environ['OPENAI_API_KEY'] = cfg["api_keys"]["openai_token"]
#     sd_model = cfg["genai_model"]["stable_diffusion"]
#     n_predictions = cfg["app_config"]["n_predictions"]
#     llama_path = cfg["genai_model"]["llama_path"]
#     return llama_path,sd_model,n_predictions

SYSTEM_PROMPT = """<s>[INST] <<SYS>>
You are a helpful bot. Your answers are clear and concise.
<</SYS>>

"""

model = "meta-llama/Llama-2-7b-chat-hf" # meta-llama/Llama-2-7b-chat-hf
tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)

llama_pipeline = pipeline(
    "text-generation",  # LLM task
    model=model,
    device="cpu" ,
    torch_dtype=torch.float32,
    # device_map="auto",
)

def format_message(message: str, history: list, memory_limit: int = 3) -> str:
    """
    Formats the message and history for the Llama model.

    Parameters:
        message (str): Current message to send.
        history (list): Past conversation history.
        memory_limit (int): Limit on how many past interactions to consider.

    Returns:
        str: Formatted message string
    """
    if len(history) > memory_limit:
        history = history[-memory_limit:]

    if len(history) == 0:
        return SYSTEM_PROMPT + f"{message} [/INST]"

    formatted_message = SYSTEM_PROMPT + f"{history[0][0]} [/INST] {history[0][1]} </s>"

    # Handle conversation history
    for user_msg, model_answer in history[1:]:
        formatted_message += f"<s>[INST] {user_msg} [/INST] {model_answer} </s>"

    # Handle the current message
    formatted_message += f"<s>[INST] {message} [/INST]"

    return formatted_message

def get_llama_response(message: str, history: list) -> str:
    """
    Generates a conversational response from the Llama model.

    Parameters:
        message (str): User's input message.
        history (list): Past conversation history.

    Returns:
        str: Generated response from the Llama model.
    """
    query = format_message(message, history)
    response = ""
    try:
        sequences = llama_pipeline(
            query,
            do_sample=False,
            # top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=256,
        )

        print("Sequences:", sequences) 


        generated_text = sequences[0]['generated_text']
        response = generated_text[len(query):] 
        print("Chatbot:", response.strip())
        return response.strip()
    except Exception as e:
        print("Error during generation:", e)
        return ""