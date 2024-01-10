from typing import Any, Dict, Iterator, Optional
from uuid import UUID
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.output import LLMResult
from langchain.schema.runnable import RunnableConfig
from threading import Thread

from queue import Queue

class StreamingHandler(BaseCallbackHandler):
    def __init__(self, queue) -> None:
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs) -> Any:
        self.queue.put(token)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        self.queue.put(None)

chat = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)

prompt = ChatPromptTemplate.from_messages([
    ('human', "{content}")
])

class StreamableChain(LLMChain):
    def stream(self, input: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        queue = Queue()
        handler = StreamingHandler(queue)

        def task():
            self(input, callbacks=[handler])
        Thread(target=task).start()
        while True:
            token = queue.get()
            if token is None:
                break
            yield token

class StreamingChain(StreamableChain, LLMChain):
    pass

chain = StreamingChain(llm=chat, prompt=prompt)

for output in chain.stream(input={'content': '自己紹介をしてくれますか？'}):
    print(output)

