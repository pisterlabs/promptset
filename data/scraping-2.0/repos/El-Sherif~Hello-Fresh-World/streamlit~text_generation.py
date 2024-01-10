from openai import OpenAI
from prompt_store import get_prompt_1, rekognition_prompt



def run_rekognition_prompt(prompt):
    client = OpenAI(api_key='sk-OPENAPIKEY')
    emotion, persona, props = prompt
    prompt_text = rekognition_prompt(emotion)
    response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"You are a helpful assistant. you are helping a guy find a recipe, the guy has a persona {persona} that consists of {props}, make sure to tell him you know these propoerties and his persona. the prompt is: {prompt_text}"}
    ]
    )
    return response.choices[0].message.content



def generate_response(*args):
        return chat(*args)
    
def chat(user_input):
    print('insde recipe generation func')
    # Your OpenAI API key
    client = OpenAI(api_key='sk-OPENAPIKEY')

    # Constructing the prompt
    prompt_text, persona, props = user_input

    response = client.chat.completions.create(
  model="gpt-4-1106-preview",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"You are a helpful assistant. you are helping a guy find a recipe, the guy has a persona {persona} that consists of {props}, make sure to tell him you know these propoerties and his persona. the prompt is: {prompt_text}"}
  ]
)

    return response.choices[0].message.content



def generate_recipe(age, gender, mood):
    print('insde recipe generation func')
    # Your OpenAI API key
    client = OpenAI(api_key='sk-OPENAPIKEY')

    # Constructing the prompt
    prompt_text = get_prompt_1(age, gender, mood)

    response = client.chat.completions.create(
  model="gpt-4-1106-preview",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"{prompt_text}"}
  ]
)

    return response.choices[0].message.content


def generate_recipe_dyn(prompt_text):
    print('insde dyn recipe generation func')
    # Your OpenAI API key
    client = OpenAI(api_key='sk-OPENAPIKEY')

    response = client.chat.completions.create(
  model="gpt-4-1106-preview",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"{prompt_text}"}
  ]
)

    return response.choices[0].message.content
