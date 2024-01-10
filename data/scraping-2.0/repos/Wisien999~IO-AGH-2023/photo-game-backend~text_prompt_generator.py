import openai
import os
import json


def get_prompts(n=5, theme=None):

    content = f"Please generate {n} unique image descriptions, each between 5 to 10 words in length. Theme is {theme}"
    if theme is None or theme == "":
        content = f"Please generate {n} unique image descriptions, each between 5 to 10 words in length."

    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": "You are programming help, your answers shouldn't include any text but json format like {\"descriptions\":  list}"},
            {"role": "user", "content": content}
        ]
    )
    json_response = json.loads(response.choices[0].message.content)
    prompt_list = json_response["descriptions"]
    # prompt_list = ['winter', 'summer', 'spring', 'autumn']
    return prompt_list