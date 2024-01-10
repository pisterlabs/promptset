import redis
import uuid
import json
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from dotenv import load_dotenv
import os

load_dotenv()

class SessionManager:
    def __init__(self):
        self.redis_client = redis.Redis(host="localhost", port=6379, db=0)

    def create_session(self):
        session_id = str(uuid.uuid4())
        self.redis_client.set(session_id, "")
        return session_id

    def get_conversation_memory(self, session_id):
        message_history_data = self.redis_client.get(session_id)
        if message_history_data is not None:
            message_history_data = message_history_data.decode("utf-8")  # Decode bytes to str
            message_history = RedisChatMessageHistory(message_history_data)
            return message_history
        else:
            return RedisChatMessageHistory(session_id=session_id)

    def getdb_connection(self):
        
        # uri = 'mysql+pymysql://aduser:adxyz123@127.0.0.1:3309/agentdesks'
        uri = self.getDbConnectionURI()
        db = SQLDatabase.from_uri(uri, include_tables=['scheduled_event_rooms', 'signup_and_login_table'], sample_rows_in_table_info=2)
        llm = OpenAI(model_name = "gpt-4", temperature=0, verbose=True)
        db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
        return db_chain

    def getDbConnectionURI(self):
        prefixString = "mysql+pymysql://"
        username = os.getenv('DBUSERNAME')
        password = os.getenv('DBPASSWORD')
        address = os.getenv('DBADDRESS')
        dbName = os.getenv('DBNAME')
        uri = prefixString+username+":"+password+"@"+address+"/"+dbName
        return uri
