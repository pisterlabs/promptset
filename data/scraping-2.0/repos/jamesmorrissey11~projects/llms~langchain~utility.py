import openai


def get_token_count(text, model_name="text-davinci-003"):
    tokenizer_model = openai.model(model_name)
    return tokenizer_model.token_count(text)
