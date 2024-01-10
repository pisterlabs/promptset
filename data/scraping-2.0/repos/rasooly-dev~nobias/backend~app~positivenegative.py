import openai
import json
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_completion(prompt, model="gpt-4"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

# RETURNS A TUPLE
# [0] = positive int value
# [1] = explanation string
def positiveNegative(givenText):

    prompt1 = f"""
    You will be provided with text delimited by triple quotes. 

    analyze this text to determine how positive or negative this text is, in terms of adjectives, events described and overall emotion
    the return value should be two numbers that add up to 100: one for the percentage of positive and one for the percentage of negative

    return the number representing only the positive percentage

    \"\"\"{givenText}\"\"\"
    """
    num = int(float(get_completion(prompt1)))

    prompt2 = f"""
    You will be provided with text delimited by triple quotes. 

    analyze this text to determine how positive or negative this text is, in terms of adjectives, events described and overall emotion
    the return value should be two numbers that add up to 100: one for the percentage of positive and one for the percentage of negative

    explain why you gave it {num} positive and {100-num} negative, without using 'I'

    \"\"\"{givenText}\"\"\"
    """
    explain = get_completion(prompt2)

    tup = (num, explain)

    return tup