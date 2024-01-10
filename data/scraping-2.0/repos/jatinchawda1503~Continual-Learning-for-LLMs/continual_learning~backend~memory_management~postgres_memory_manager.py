from datetime import datetime
from langchain.schema import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage,message_to_dict,messages_from_dict

import json
import psycopg
from psycopg.rows import dict_row

from typing import List

class PostGresMemoryHistory(BaseChatMessageHistory):
    def __init__(self, db_name:str, db_user:str, db_password:str, db_host:str, db_port:str, table_name:str, chat_id:str):
        self.db_config = {
            'dbname': db_name,     
            'user': db_user,        
            'password': db_password,    
            'host': db_host,           
            'port': db_port,
            'table_name': table_name,                 
        }
        
        self.chat_id = chat_id
        
        
        try:
            self.conn = psycopg.connect(dbname=self.db_config['dbname'], user=self.db_config['user'], password=self.db_config['password'], host=self.db_config['host'], port=self.db_config['port'])
            self.cur = self.conn.cursor(row_factory=dict_row)
            self.create_table_if_not_exists()
        except psycopg.Error as e:
            print("Error connecting to PostgreSQL database:", e)
            self.conn = None
            self.cur = None
                  
    
    def create_table_if_not_exists(self):
        create_table_query = f"""CREATE TABLE IF NOT EXISTS {self.db_config['table_name']} (
            id SERIAL PRIMARY KEY,
            chat_id TEXT NOT NULL,
            message JSONB NOT NULL, 
            time_stamp TIMESTAMP NOT NULL
        );"""
        self.cur.execute(create_table_query)
        self.conn.commit()
    
    def add_ai_message(self, message: str) -> None:
        return self.add_message(AIMessage(content=message))
    
    def add_user_message(self, message: str) -> None:
        return self.add_message(HumanMessage(content=message))
    
    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from PostgreSQL"""
        query = (
            f"SELECT * FROM {self.db_config['table_name']} WHERE chat_id = %s ORDER BY time_stamp;"
        )
        self.cur.execute(query, (self.chat_id,))
        items = [record["message"] for record in self.cur.fetchall()]
        messages = messages_from_dict(items)
        return messages
    
    
    def add_message(self, message:BaseMessage):
        time_stamp = datetime.now()
        query = f"INSERT INTO {self.db_config['table_name']} (chat_id, message, time_stamp) VALUES (%s, %s, %s);"
        self.cur.execute(query, (self.chat_id, json.dumps(message_to_dict(message)), time_stamp))
        self.conn.commit()
        
            
    def clear(self) -> None:
        conn = psycopg.connect(self.db_config)
        cur = conn.cursor()
        if conn:
            try:
                query = f"DELETE FROM {self.db_config['dbname']} WHERE session_id = %s;"
                cur.execute(query, (self.session_id,))
                conn.commit()
                cur.close()
                conn.close()
            except psycopg.Error as e:
                print("Error clearing DB:", e)
        else:
            print("Failed to establish connection...")