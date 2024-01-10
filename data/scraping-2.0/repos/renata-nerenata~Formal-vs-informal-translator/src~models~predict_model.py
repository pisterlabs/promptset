import openai

openai.api_key = "you_key"


def inference_pipeline(text):
    return openai.Completion.create(
        model="babbage:ft-mcgill-2022-12-15-12-39-35", prompt=text
    )["choices"][0]["text"].replace("\n\n###\n\n", "")
