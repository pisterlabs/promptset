from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from database import get_db_connection
from langchain.schema.messages import HumanMessage, AIMessage


class StreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, container, initial_text="", display_method="markdown"):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")


# Modify the StreamlitChatMessageHistory to save and load from the database
class StreamlitChatMessageHistoryDB(StreamlitChatMessageHistory):
    def __init__(self, key, conversation_id=None):
        super().__init__(key=key)
        self.conversation_id = conversation_id

    def init_conversation(self):
        if self.conversation_id is None:
            self.conversation_id = self.start_new_conversation("Daily chat")

    def start_new_conversation(self, title):
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO conversations (title) VALUES (?)", (title,))
        conn.commit()
        conversation_id = cur.lastrowid
        conn.close()
        return conversation_id

    def add_message(self, type, content):
        self.init_conversation()
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO messages (conversation_id, type, content) VALUES (?, ?, ?)",
            (self.conversation_id, type, content),
        )
        conn.commit()
        conn.close()

    def add_user_message(self, content, new_message=True):
        self._messages.append(HumanMessage(content=content))
        if new_message:
            self.add_message("human", content)

    def add_ai_message(self, content, new_message=True):
        self._messages.append(AIMessage(content=content))
        if new_message:
            self.add_message("ai", content)

    def load_history(self):
        conn = get_db_connection()
        messages = conn.execute(
            "SELECT type, content FROM messages WHERE conversation_id = ? ORDER BY timestamp",
            (self.conversation_id,),
        ).fetchall()
        # print(messages)
        for msg in messages:
            if msg["type"] == "human":
                self.add_user_message(msg["content"], False)
            else:
                self.add_ai_message(msg["content"], False)
        conn.close()
