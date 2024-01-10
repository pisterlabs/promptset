import os
import openai
from convlab2.nlg.generative_models.azure_openai import AzureOpenAIModel


def test_azure_opeanai():
    openai.api_type = "azure"
    openai.api_version = "2023-03-15-preview"
    openai.api_base = os.getenv("OPENAI_API_BASE")  # Your Azure OpenAI resource's endpoint value.  # noqa
    assert openai.api_base, "Need OPENAI_API_BASE env var"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    assert openai.api_key, "Need OPENAI_API_KEY env var"

    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo",  # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.  # noqa
        messages=[
            {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},  # noqa
            {"role": "user", "content": "What's the difference between garbanzo beans and chickpeas?"}  # noqa
        ]
    )
    print(response)

    answer = response['choices'][0]['message']['content']
    print(answer)

    assert answer


def test_prepare_prompt():

    AzureOpenAIModel._prepare_prompt(
        'Goal: Conversation: ex1 Conversation: ex2 Conversation: ongoing')
