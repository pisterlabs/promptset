# a collection of LLMs and their simplified APIs, for now mostly open AI chatgpt
import openai
from googles import load_json



if __name__ == '__main__':

    secrets = load_json('./secrets_openai.json')
    openai.api_key = secrets['api-key']
    # openai.organization = "gender-project"
    model_list = openai.Model.list()

    print(model_list)