import openai
import os

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.getenv('OPEN_AI_API_KEY')


def get_completion(prompt,
                   model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


if __name__ == '__main__':
    prod_review = """
    Got this panda plush toy for my daughter's birthday,
    who loves it and takes it everywhere. It's soft and  
    super cute, and its face has a friendly look. It's 
    a bit small for what I paid though. I think there
    might be other options that are bigger for the
    same price. It arrived a day earlier than expected, 
    so I got to play with it myself before I gave it
    to her.
    """

    prompt = f"""
    Your task is to generate a short summary of a product
    review from an ecommerce site. 
    
    Summarize the review below, delimited by triple 
    backticks, in at most 30 words. 
    
    Review: ```{prod_review}```
    """

    response = get_completion(prompt)
    print(response)
