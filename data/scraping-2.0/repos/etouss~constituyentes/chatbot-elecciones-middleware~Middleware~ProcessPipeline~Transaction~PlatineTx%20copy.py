from ..Conversation import Conversation
from ..GenericTransaction import GenericTransaction
from ..TransactionChain import TransactionChain
from ..TransactionChainChat import TransactionChainChat
import psycopg2

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


class PlatineTx(GenericTransaction):
    def __init__(self, conversation:Conversation) -> None:
        super().__init__('MarceloTxChat', conversation)

    def process_input(self, input:str) -> str:
        instruction = """Your task is to design an SQL query to filter the tuples of the databases which are relevant to answer a user question. The SQL query should filter based on other metadata associated with each tuple, such as the date or name(s). 
        Using the resulting tuples, we will then run a semantic similarity tool to identify the tuple(s) with the content most relevant to the user's question.

You need to split the user question into two parts:
1. An SQL query that filters tuples based on metadata only.
2. A question that use the tuples to answer user's.\n\n"""
        schema = """Database schema: 
CREATE TABLE tweets_meta (int tweet_id PRIMARY KEY, date date, user_real_name text, username text, likes integer, retweets integer, is_retweet integer);\n
"""

        db = psycopg2.connect(host="db-postgresql-nyc1-26343-do-user-9263663-0.b.db.ondigitalocean.com", 
            port="25060",
            database="elections",
            user="jreutter",
            password="AVNS_UM4p-njPfaQ6q823wvN")
        cur = db.cursor()
        cur.execute("SELECT * FROM tweets LIMIT 3")
        rows = cur.fetchall()
        data = "Database tuples:\n"
        for row in rows:
            #print(row)
            data += str(row) + "\n"
        cur.close()
        query = "Return the two questions following the format:\n SQL query = 'SELECT ...' \n Question = 'Give a summary about electrity and ...?'"

        #history = self.conversation.get_history()

        #txc = TransactionChain(self.tx_id)
        txc = TransactionChainChat(self.tx_id)
        
        messages = [
            SystemMessage(content=instruction + schema + data + query),
            HumanMessage(content=input)
        ]
        #You can modifify SystemMessage and HumanMessage. 
        #Normally people put instruction in system message and input in Humman message.

        #Have Fun 

        print(instruction + schema + data + query)
        print(input)
        answer = txc.process_input(messages)
        print(answer)

        #cur = db.cursor()
        #cur.execute(sql_query)
        #rows = cur.fetchall()
        #sql_results = "Database tuples:\n"
        #for row in rows:
            #print(row)
        #    sql_results += str(row) + "\n"
        #cur.close()

        return answer

        

        
        #for tx_id,input_h,output_h in history:
        #    new_input += 'USER: '+input_h+'\n'
        #    new_input += 'ASSISTANT: '+output_h+'\n'
        #new_input += 'USER: '+input+'\n'
        #new_input += 'ASSISTANT: '

        #print(new_input)

        
        