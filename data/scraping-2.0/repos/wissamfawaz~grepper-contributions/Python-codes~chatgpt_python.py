# Reference: https://github.com/openai/openai-python
# $pip install openai
from os import getenv
import openai

# Set the api key
openai.api_key = getenv("OPENAI_API_KEY")

def get_models():
    """
    This function returns a list of all the language
    models offered by openai.
    """
    return openai.Model.list()

def print_model_ids(retrieved_models):
    """
    This function stores the model ids in a string
    and then prints that string out
    """
    model_ids = ""
    models_data = retrieved_models.data
    for idx in range(0, len(retrieved_models.data)):
        model_ids += models_data[idx].id + " "
    print(model_ids)

# Create a list of all supported models
models = get_models()

# print all model ids out
print_model_ids(models)
