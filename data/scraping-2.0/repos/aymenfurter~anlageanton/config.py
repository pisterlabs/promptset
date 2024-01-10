import os
from langchain.chat_models import AzureChatOpenAI

def load_configuration():
    deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
    model_name = os.getenv("AZURE_MODEL_NAME")
    temperature = 0.4

    llm = AzureChatOpenAI(
        deployment_name=deployment_name,
        model_name=model_name,
        temperature=temperature,
    )

    return dict(
        llm=llm,
        verbose=False,
        conversation_history=[],
    )