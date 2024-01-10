import openai
import os
import dotenv

dotenv.load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')


def get_response(prompt):
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=prompt,
      max_tokens=2500
    )
    return response.choices[0].text.strip()




