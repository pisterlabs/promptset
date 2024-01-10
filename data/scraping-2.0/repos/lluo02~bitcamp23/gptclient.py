import openai,os,sys
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('API_KEY')

def generate_response(message):
    prompt = f"User: {message}\nChatGPT:"
    response = openai.ChatCompletion.create(
      model='gpt-3.5-turbo',
      messages = [{"role": "system", "content": "You are a human being who does not want to engage or be bothered."},
                  {"role": "user", "content": message}],
      max_tokens=1024,
      n=1,
      stop=None,
      temperature=0.7,
    )
    message = response.choices[0].message.content
    return message




def generate_friend_response(message):
    prompt = f"User: {message}\nChatGPT:"
    response = openai.ChatCompletion.create(
      model='gpt-3.5-turbo',
      messages = [{"role": "system", "content": "You are a college student being texted by a friend. Answer with a text message. Do not answer homework questions."},
                  {"role": "user", "content": message}],
      max_tokens=1024,
      n=1,
      stop=None,
      temperature=0.7,
    )
    message = response.choices[0].message.content
    return message

def generate_parent_response(message):
    prompt = f"User: {message}\nChatGPT:"
    response = openai.ChatCompletion.create(
      model='gpt-3.5-turbo',
      messages = [{"role": "system", "content": "Answer with a text message, but do not say anything that might make the recipient worry, because they are family."},
                  {"role": "user", "content": message}],
      max_tokens=1024,
      n=1,
      stop=None,
      temperature=0.7,
    )
    message = response.choices[0].message.content
    return message