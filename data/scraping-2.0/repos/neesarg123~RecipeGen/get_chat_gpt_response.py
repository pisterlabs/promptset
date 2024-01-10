import openai
import os
from create_query import generate_gpt_query
from utils import write_string_to_file, get_title_of_recipe


openai.api_key = os.getenv("OPENAI_API_KEY") 


def get_gpt_response():
    query = generate_gpt_query()

    response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f'{query}'},
            ]
        )

    return response["choices"][0]["message"]["content"]


gpt_response = get_gpt_response()
print(gpt_response)
# write response to file
write_string_to_file(text=gpt_response, filename=get_title_of_recipe(gpt_response))

