import os
from tenacity import retry, wait_random_exponential, stop_after_attempt
import openai
import dotenv
dotenv.load_dotenv('.env')
from models.models import Message, MessageRole

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_TEMPERATURE = 0.0
DEFAULT_CHAT_MODEL = "gpt-3.5-turbo" # "gpt-4"

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_chat_completion(
    prompt: str, 
    messages: list[Message], 
    model: str = DEFAULT_CHAT_MODEL, 
    temperature: float = MODEL_TEMPERATURE
):
    """Returns response to the given prompt."""
    system_message = [{"role": MessageRole.system, "content": prompt}]
    message_dicts = [{"role": message.role, "content": message.text} for message in messages]
    conversation_messages = system_message + message_dicts
    response = openai.ChatCompletion.create(
        model=model,
        messages=conversation_messages,
        temperature=temperature
    )
    return response.choices[0]['message']['content'].strip()

def setup_prompt(
    prompt_file: str = 'prompts/summarize_prompt.md', 
    replacement_content: str = None, 
    replacement_string: str = None
) -> str:
    """Creates a prompt for generating a response."""
    with open(prompt_file) as f:
        prompt = f.read()
        if replacement_content and replacement_string:
            prompt = prompt.replace(replacement_string, replacement_content)
    return prompt