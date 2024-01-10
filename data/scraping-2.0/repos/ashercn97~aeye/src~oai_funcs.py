from openai import OpenAI
import os
import openai


openai.api_key = os.getenv('OPENAI_API_KEY')


# class of different system prompts, useful
class SystemPrompts:
    TYPE_CHECKER = """You are a natrual language "type" checker. This means that, when given a description of a "type" such as 'one word' and an item such as 'dog', you will return the term "True". If you were given the description 'one name' and the item you recieved was 'cat', you would return "False". Your job is to figure out whether the descriptions match the item, and reply with only a True or False. You will be given prompts in the format: 
    DESCRIPTION: "des"
    ITEM: "item"
    
    and you will respond in the format:
    
    True/False
     
    You will only rpely with the word True or the word False, nothing else. """

class UserPrompts:
    TYPE_CHECKER = """ 
    DESCRIPTION: {}
    ITEM: {} """



# setups the client
def setup_client():
    # defaults to getting the key using os.environ.get("OPENAI_API_KEY")
    client = OpenAI()
    return client


# gets a completion given lots of info
def get_completion(client, message, system_prompt, json=False, model="gpt-3.5-turbo-1106"):
    if json:
        completion = client.chat.completions.create(
            model=model,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
        )
    else:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
        )

    return completion.choices[0].message


