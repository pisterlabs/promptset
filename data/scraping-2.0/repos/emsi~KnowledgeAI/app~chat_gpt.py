from typing import Any

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

import chat
from config import settings


class Chat(chat.Chat):
    """Chatbot class. Implemented using OpenAI API."""

    def __init__(self, stream_container):
        self.response = ""

        class StreamingStreamlitOutCallbackHandler(StreamingStdOutCallbackHandler):
            chat = self

            def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
                """Run on new LLM token. Only available when streaming is enabled."""
                self.chat.response += token
                stream_container.markdown(self.chat.response)

        self.chat = ChatOpenAI(
            model_name=settings.MODEL,
            temperature=settings.TEMPERATURE,
            openai_api_key=settings.OPENAI_API_KEY,
            streaming=True,
            callbacks=[StreamingStreamlitOutCallbackHandler()],
        )

    def __call__(self, question, docsearch):
        """Ask a question to the chatbot."""
        if settings.TOP_K:
            documents = docsearch.similarity_search(question, k=settings.TOP_K)
            document = "\n\n".join([doc.page_content for doc in documents])
            messages = [
                SystemMessage(content=settings.SYSTEM_PROMPT),
                HumanMessage(
                    content=f'Given following document, please answer following question: "{question}"? '
                    f'\n\nDOCUMENT: ```{document}```\n\nEND OF DOCUMENT\n\nQUESTION: "{question}"\n\n'
                ),
            ]
        else:
            messages = [
                SystemMessage(content=settings.SYSTEM_PROMPT),
                HumanMessage(content=question),
            ]
        response = self.chat(messages)
        self.log(question, response.content, messages[1])
