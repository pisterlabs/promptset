import pathlib

import cachetools
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory


class ChatHistoryHelper:
    _path_history = pathlib.Path(__file__).parent.parent.resolve() / 'chat-history'
    memory_key = 'chat_history'

    def __init__(self, path_history=None):
        if path_history:
            self._path_history = path_history

    @cachetools.cached(cache={})
    def _get_history(self, user_id):
        path = self._path_history / f"{user_id}.json"
        return FileChatMessageHistory(str(path))

    def write_history(self, user_id, message):
        self._get_history(user_id).add_message(message)

    def get_memory(self, user_id):
        history = self._get_history(user_id)
        memory = ConversationBufferMemory(
            memory_key=self.memory_key,
            chat_memory=history,
        )
        return memory