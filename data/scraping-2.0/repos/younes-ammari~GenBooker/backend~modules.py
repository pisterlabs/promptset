
# # OpenAI key from environment
from langchain.llms import OpenAI


openai_api_key = 'your_openai_api_key'


class initializeBot():
    def __init__(self) -> None:
        self.model = OpenAI(openai_api_key=openai_api_key)
        pass


    def chat(self, message):
        return self.model.predict(message)


