import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key = os.getenv('OPENAI_API_KEY')

def get_completion(prompt, model="gpt-3.5-turbo", temperature=0): 
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
    )
    return response.choices[0].message["content"]

movies = [
  "Cars",
  "Frozen",
]

for movie in movies:
  prompt = f"""
    Write a movie review for the movie below identified between 
    three sets of backticks in the same pithy and cynical tone
    as the example below. If there is a remake of the movie,
    review the original.

    Example movie: "The Wizard of Oz"
    Example review: "The Wizard of Oz (1939): Transported to a 
    surreal landscape, a young girl kills the first person she
    meets then teams up with three strangers to kill again."

    ```{movie}```
  """

  print(f"{get_completion(prompt)}\n\n")