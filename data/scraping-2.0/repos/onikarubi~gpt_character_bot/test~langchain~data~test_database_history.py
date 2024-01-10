from app.apis.openai.gpt.langchains.memory.database_chat_history import PostgresChatHistory
from langchain.memory import PostgresChatMessageHistory
import os

class TestPostgresChatHistory:
    target_history_object: PostgresChatHistory = PostgresChatHistory(session_id='test_session')

    def test_load_history(self):
        connect_db = f'{self.target_history_object.database_name}://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@{os.getenv("POSTGRES_HOST")}:{os.getenv("POSTGRES_PORT")}'
        test_history = PostgresChatMessageHistory(session_id=self.target_history_object.session_id, connection_string=connect_db)
        assert test_history.messages == self.target_history_object.load_history().messages
