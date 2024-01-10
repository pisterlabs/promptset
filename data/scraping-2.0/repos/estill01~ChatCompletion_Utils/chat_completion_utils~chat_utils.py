import openai
import os
import tiktoken
from typing import List, Dict

openai.api_key = os.environ.get("OPENAI_API_KEY")

MODELS = {
    'models': ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-32k'],
    'gpt-3.5-turbo': {
        'model': 'gpt-3.5-turbo',
        'max_tokens': 4096,
    },
    'gpt-4': { 
        'model': 'gpt-4',
        'max_tokensi': 8192,
    },
    'gpt-4-32k': {
        'modeli': 'gpt-4-32k',
        'max_tokens': 32768,
    },
}

DEFAULT_MODEL=MODELS['gpt-4']['model']

def set_default_model(model_family: str='gpt-4') -> None:
    DEFAULT_MODEL=model_family

# ---------------------------------------------------------

def llm(
    system_instruction: str = None, 
    user_input: str = None, 
    messages: List[Dict[str, str]] = [], 
    temp: int = 0, 
    model_family: str = DEFAULT_MODEL,
) -> str:
    messages = _build_prompt(
        system=system_instruction,
        user=user_input,
        messages=messages,
    )
    model = select_model(messages, DEFAULT_MODEL)
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temp,
        max_tokens=MODELS[model].max_tokens,
    )
    # TODO Add (optional) output formatter here before returning value
    return response.choices[0].message.content


# ----------------------------------------------------
# PROMPT PARTIALS
# ----------------------------------------------------

# Add language to prompts to help steer LLM output

def _code_prompt(prompt: str = None) -> str:
    # TODO Check if received `prompt` ends with a '.' and accomodate
    base_prompt = f"Generate code. Only output code. Do NOT include introductory text or explain the code you have generated after producing it. If you are unable to perform the code translation respond 'ERROR'"
    return f"{prompt} {base_prompt}"


# ----------------------------------------------------
# PROMPT CONSTRUCTION
# ----------------------------------------------------

# SOURCE: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def select_model(messages: List[Dict[str, str]], model_family: str = DEFAULT_MODEL, force: bool = False) -> str:
    msg_count = len(messages)
    token_count = num_tokens_from_messages(messages, model_family)
    avg_tokens_message = int(token_count / msg_count)
    count = token_count + avg_tokens_message

    if model_family == MODELS['gpt-3.5-turbo'].model:
        if count < MODELS['gpt-3.5-turbo'].max_tokens or force:
            return MODELS['gpt-3.5-turbo'].model
        else:
            raise ValueError(f"{MODELS['gpt-3.5-turnbo'].model}: Token count ({count}) greater than max allowed tokens ({MODELS['gpt-3.5-turbo'].max_tokens}).")
    if count < MODELS['gpt-4'].max_tokens: 
        return MODELS['gpt-4'].model
    elif count < MODELS['gpt-4-32k'].max_tokens or force:
        return MODELS['gpt-4-32k'].model
    else:
        raise ValueError(f"{MODELS['gpt-4-32k'].model}: Token count ({count}) greater than max allowed tokens ({MODELS['gpt-4-32k'].max_tokens}).")

def _build_user_prompt(content: str) -> Dict[str, str]:
    return {'role': 'user', 'content': conent }

def _build_system_prompt(content: str) -> Dict[str, str]:
    return {'role': 'system', 'content': content }

def build_prompt(
    *, 
    system_content: str = None, 
    user_content: str = None, 
    messages: List[Dict[str, str]] = [], 
    ) -> List[Dict[str, str]]:

    if system_content is not None and user_content is not None:
        return messages.extend([_make_system_prompt(system_content), _make_user_prompt(user_content)])
    if system_content is None and user_content is not None:
        return messages.extend([_make_user_prompt(user_content)])
    else:
        raise ValueError("Must provide `user_content`")

