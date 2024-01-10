"""
LLM API Interface.

This module provides an interface to interact with LLM using the HuggingFace Endpoint.
It supports token counting, memory reset, and conversation continuation.
"""

import json
import requests
import toml
import tiktoken

from langchain.llms import HuggingFaceEndpoint
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate

# Load secrets from the configuration file
CONFIG = toml.load(".streamlit/secrets.toml")
BEARER_TOKEN = CONFIG["huggingface"]["bearer"]
HG_API_TOKEN = CONFIG["huggingface"]["api_token"]
ENDPOINT_URL = CONFIG["huggingface"]["endpoint_url"]

# Define system limits
MAX_SYS_TOKENS = 256  # Max tokens for system message
MAX_MODEL_TOKENS = 600  # Max tokens for LLM input

# Initialize HuggingFace endpoint
hf = HuggingFaceEndpoint(
    task="text-generation",
    endpoint_url=ENDPOINT_URL,
    model_kwargs={"max_new_tokens": 256},
    huggingfacehub_api_token=HG_API_TOKEN
)

memory_template = """[INST] <<SYS>>
Progressively summarize the lines of conversation provided, adding onto the previous summary 
returning a new summary.

EXAMPLE
Current summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.
                          
New lines of conversation:
Human: Why do you think artificial intelligence is a force for good?
AI: Because artificial intelligence will help humans reach their full potential.

New summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
END OF EXAMPLE
<</SYS>>

Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:[/INST]"""
memory = None


def reset_memory():
    """Resets the memory of the LLM."""
    global memory
    memory = ConversationSummaryBufferMemory(
        llm=hf, max_token_limit=MAX_MODEL_TOKENS - MAX_SYS_TOKENS, return_messages=True,
        prompt=PromptTemplate(input_variables=['summary', 'new_lines'],
                              template=memory_template))


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Calculate the number of tokens in a text string.

    Args:
        string (str): The input string.
        encoding_name (str): The name of the encoding to use.

    Returns:
        int: Number of tokens in the string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))


def continue_conversation(messages, system_message="", max_new_tokens=512, temperature=0.7):
    """
    Continue the conversation using the LLM.

    Args:
        messages (list): A list of previous messages.
        system_message (str, optional): System message to guide the LLM. Defaults to "".
        max_new_tokens (int, optional): Max new tokens for LLM response. Defaults to 512.
        temperature (float, optional): Sampling temperature. Defaults to 0.7.

    Returns:
        str: LLM response.
    """
    # Initialize the prompt with Llama2 formatting
    prompt_template = f'''[INST] <<SYS>>\n{system_message}\n'''

    # Add messages to history when there are at least 3 messages
    if len(messages) > 2:
        memory.save_context({"input": messages[-3]["content"]}, {"output": messages[-2]["content"]})

    # Add summary of conversations to prompt (can be empty string if buffer is not full)
    prompt_template += memory.moving_summary_buffer + "<</SYS>>\n\n"

    # Add messages in buffer to prompt
    for curr_message in memory.chat_memory.messages:
        prefix = ""
        suffix = ""
        if isinstance(curr_message, HumanMessage):
            prefix = "[INST]"
            suffix = "[/INST]"

        prompt_template += f"\n{prefix}{curr_message.content}{suffix}"

    # Add latest message to prompt
    prompt_template += f"\n[INST]{messages[-1]['content']}[/INST]"

    # Debugging info
    print(f"Prompt tokens: {num_tokens_from_string(prompt_template, encoding_name='cl100k_base')}")
    print(prompt_template + "\n\n")

    # Define the payload
    payload = {
        "inputs": prompt_template,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": 50,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "batch_size": 1,
            "stop": ["</s>", "[INST]", "[/INST]"]  # For models that take them
        }
    }

    # Define the headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {BEARER_TOKEN}'
    }

    # Make the POST request
    try:
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload)
        response.raise_for_status()
        json_response = json.loads(response.text)
        model_response = json_response[0]['generated_text']
    except requests.RequestException:
        model_response = "Can you retry please?"

    return model_response
