from langchain.callbacks.base import BaseCallbackHandler

from api.llm.history import histories
from api.llm.models import conversation_retrieval_chain


class QueueCallback(BaseCallbackHandler):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.queue.put(token)

    def on_llm_end(self, *args, **kwargs):
        return self.queue.empty()


async def conversational(prompt: str, user_id: str, chat_id: str, queue):
    chat_history = histories.get(user_id, chat_id)

    response = await conversation_retrieval_chain.acall(
        {"question": prompt, "chat_history": chat_history},
        callbacks=[QueueCallback(queue)],
    )

    answer = response["answer"]
    histories.append(user_id, chat_id, (prompt, answer))
