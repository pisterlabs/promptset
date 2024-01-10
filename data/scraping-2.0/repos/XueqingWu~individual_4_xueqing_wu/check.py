import os

# from dotenv import load_dotenv
import openai

openai.api_key = os.getenv("API_TOKEN")


def get_completion(prompt, model="gpt-3.5-turbo"):
    prompt_answer = f"""
    Perform the following actions: 

    1 - I will give an illness I have
    2 - Provide the list of nutritions I need to take
    3 - Give only the list of the nutritions, without the food, 
    description, or function of the nutrition
    4 - For the nutritrion part, I want it in this format:

    Using the following format:
    Illness: <Illness name>
    Nutrition: <Nutrition list>
   

    ```{prompt}```
    """

    messages = [{"role": "user", "content": prompt_answer}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]
