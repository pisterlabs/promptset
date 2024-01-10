from langchain.prompts.chat import ChatPromptTemplate

class ChatPromptTemplate(ChatPromptTemplate):
    @property
    def _prompt_type(self) -> str:
        return "chat"
