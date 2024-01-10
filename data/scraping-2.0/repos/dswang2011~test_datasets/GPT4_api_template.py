import os
import time
import openai
from azure.identity import AzureCliCredential
import tiktoken

if "openai.azure.com" not in os.environ["no_proxy"]:
    os.environ["no_proxy"]=os.environ["no_proxy"]+",openai.azure.com"

os.environ["http_proxy"]="proxy.jpmchase.net:10443"
os.environ["https_proxy"]="proxy.jpmchase.net:10443"

venv_name = "openai"  # change as needed
os.environ["PATH"] = os.environ["PATH"] + f":/opt/omniai/work/instance1/jupyter/venvs/{venv_name}/bin"

credential = AzureCliCredential()
openai_token = credential.get_token("https://cognitiveservices.azure.com/.default")
openai.api_key = openai_token.token
openai.api_base = "https://llmopenai.jpmchase.net/WS0001037P-exp" #required
openai.api_type = "azure_ad" # required
openai.api_version = "2023-05-15" # change as needed

# Load the GPT-3.5 tokenizer
# Initialize the tiktoken tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
gpt_version = "gpt-4-0613"
tokenizer = tiktoken.get_encoding("cl100k_base")


def truncate_prompt(input_prompt, max_tokens=8192, error_margin=100):
    input_prompt_tokens = tokenizer.encode(
        input_prompt
    )
    # Calculate the number of tokens in the input prompt
    num_tokens = len(
        input_prompt_tokens
    )
    # Check if the input prompt exceeds the maximum token limit
    if num_tokens > max_tokens:
        # If it exceeds, you need to truncate it
        num_tokens_to_keep = max_tokens - error_margin  # Reserve k tokens for the model's response
        # NOTE: we pick 500 to give a lot of room for error in the length computation
        truncated_input_tokens = (
            input_prompt_tokens[:num_tokens_to_keep]
        )
        # Decode the truncated input
        truncated_input_text = tokenizer.decode(
            truncated_input_tokens
        )
        return truncated_input_text
    else:
        # If it doesn't exceed the limit, you can use the original input prompt as-is
        return input_prompt


def get_completion(doc, question):
    prompt = f"""Based on the given Document, {question} 
Please provide only the answer (no decorations or explanations). Extract from the document when possible.
Use the following format to answer:
Answer:<answer>
Document:```{doc}```
"""
    margin = 100
    while True:
        try:
            messages = [{"role": "user", "content": truncate_prompt(prompt, error_margin=margin)}]
            response = openai.ChatCompletion.create(
                engine=GPT_VERSION, 
                messages=messages,
                temperature=0
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            error_message = str(e).lower()
            if 'rate limit' in error_message and 'second' in error_message:
                print('=> sleep and retry')
                time.sleep(10)
            elif "maximum context length" in error_message:
                print('=> not enough tokens for prompt + answer')
                margin += 100
                continue
            else:
                return


