from langchain import OpenAI, ConversationChain
from langchain.chat_models import AzureChatOpenAI
import os


def chat():
    model = AzureChatOpenAI(
        model_name="gpt-35-turbo",
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        openai_api_base=os.environ["AZURE_OPENAI_API_BASE"],
        openai_api_version="2023-06-01-preview",
        deployment_name=os.environ["AZURE_OPENAI_MODEL"],
        temperature=0,
    )
    conversation = ConversationChain(llm=model, verbose=True)

    output = conversation.run("私は日本人です。何を聞いても日本語で答えてください。語尾には「にゃ。」をつけてください。")
    output = conversation.run("Who is the prime minister of Japan?")

    print(output)
