from langchain.llms import OpenAI 
from langchain.chat_models import ChatOpenAI  
from langchain.callbacks import get_openai_callback
from langchain.schema.messages import HumanMessage, SystemMessage

def get_content_from_file(file_path):
    """Read and return content from a file."""
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"No such file: '{file_path}'")
    except Exception as e:
        raise e

# Path to the API key and prompt text files
api_key_file_path = "key.txt"
prompt_file_path = "prompt.txt"

# Read API key and prompt from files
api_key = get_content_from_file(api_key_file_path)
prompt = get_content_from_file(prompt_file_path)

print("===========PROMPT: ==========\n"+prompt+"\n=============\n")

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(content=prompt)
]

chat = ChatOpenAI(
    model_name='gpt-3.5-turbo',
    openai_api_key = api_key
)

with get_openai_callback() as cb:
    chat.invoke(messages)
    for chunk in chat.stream(messages):
        print(chunk.content, end="", flush=True)
    print("\n===========CALLBACK: ==========\n")
    print(cb)
    print("\n=============\n")
