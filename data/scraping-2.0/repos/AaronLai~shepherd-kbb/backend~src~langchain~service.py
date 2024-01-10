from fastapi import UploadFile

import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain


from backend.config import Settings

class LangchainService():
    def __init__(self, settings: Settings):
        os.environ['OPENAI_API_KEY'] = settings.OPENAI_API_KEY

        llm = ChatOpenAI(
            temperature=0.1,
            model_name="gpt-3.5-turbo-16k-0613"
        )

        self.chain = ConversationChain(llm=llm)

    def runGPT(self, prompt: str):
        return self.chain.run(prompt)
