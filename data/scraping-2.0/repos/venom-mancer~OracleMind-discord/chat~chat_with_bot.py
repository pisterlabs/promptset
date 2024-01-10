import openai
import json

with open('config.json') as user_file:
  file_contents = user_file.read()
parsed_json = json.loads(file_contents)

openai.api_key = parsed_json["open_ai_token"]
openai.Model.list()


def talktobot(chat):


    response = openai.Completion.create(

        engine="text-davinci-003",
        prompt = "<|endoftext|>"+chat+"\n--\nLabel:",
        temperature=0.8,
        top_p=0.9,
        max_tokens=750,
    )

    output_label = response["choices"][0]["text"]

    return output_label