from langchain.chat_models import ChatOpenAI
from tools.secret_squirrel import SecretSquirrel

class ChatGptLlm():

    def __init__(self, model=None, temperature=0.3) -> None:
        self._model = model
        self._creds = SecretSquirrel().stash
        self._llm = ChatOpenAI(
                model_name=self._model,
                openai_api_key=self._creds['open_ai_api_key'],
                temperature=temperature
        )


    def query(self, prompt) -> str:
        return self._llm.predict(prompt)