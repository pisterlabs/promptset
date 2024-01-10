import os
import openai
from dotenv import load_dotenv

load_dotenv()


GPT3_API_KEY = os.getenv("GPT3_API_KEY")

openai.api_key = GPT3_API_KEY 

class gpt3_monologue_generator:

  def __init__(self) -> None:
     pass

  def generate_monologue(self, prompt):
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=prompt,
      temperature=0.8,
      max_tokens=500,
      # top_p=1.0,
      frequency_penalty=0.5,
      presence_penalty=0.0
    )["choices"][0]["text"]
    # fo = open("generatedTexts/gpt3_generated_text.txt","a")
    # fo.write("\nGPT3 from " + self.prompt + ": " + response)
    # fo.close()
    return response

    