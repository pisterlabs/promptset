from dotenv import load_dotenv
import os
import openai
import tiktoken

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def num_tokens_from_string(string: str) -> int:
    # """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

# print(num_tokens_from_string(read_file_to_string(filename_filtered)))

# print(filename_filtered)