import openai
import config


class ModelType:
    GPT_3_5 = "gpt-3.5-turbo",
    GPT_4 = "gpt-4",


openai.api_key = config.GPT_API_KEY
query = input("GPT Query:")
