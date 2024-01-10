from openai import Completion, api_key
import creds

api_key = creds.OPENAI_KEY

def text_io(input):

    resp = Completion.create(
            model="text-davinci-003",
            prompt=input,
            max_tokens=100,
            temperature=.9
            # stream=True <-- creates generator object
        )

    output_text = resp["choices"][0]["text"]
    return output_text
