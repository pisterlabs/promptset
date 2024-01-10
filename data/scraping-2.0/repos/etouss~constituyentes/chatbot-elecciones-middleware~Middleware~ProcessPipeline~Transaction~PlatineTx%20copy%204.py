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
        instruction = """Given a database schema your task is to seperate the user input in two.
The first part should be an SQL query using the metadata and only the metadata to lightly filter out the tuple irrelevant to the user input without using the content.
The second part should be a question using the content of the returned tuple to answer user input.
Do not try to answer the user input, your task is to design an SQL query.\n\n"""
        
        
        schema = """Database schema: 
CREATE TABLE tweets_metadata (int tweet_id PRIMARY KEY, date date, user_real_name text, username text, likes integer, retweets integer, is_retweet integer);\n
CREATE TABLE tweets_content (int tweet_id, content text, FOREIGN KEY(tweet_id));\n
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
    
        query = "The ouput should be of the format: SQL Query = 'SELECT content, .. FROM ... WHERE ...'\n"
        #query = "Return the two output following the format:\n SQL query = 'SELECT content, content ...' \n Task = 'Give a summary about electrity and ...?'"


        data = """A database sample: 
tweets_metadata: (8923, datetime.date(2022, 5, 6), 'Gloria Hutt Hesse', 'GloriaHutt', 11, 0, 0)
tweets_content: (8923, '@louisdegrange @metrodesantiago Gran foto! La citrola y el Peugeot 404...símbolos de los 70!!')\n\n"""

        #history = self.conversation.get_history()

        #txc = TransactionChain(self.tx_id)
        txc = TransactionChainChat(self.tx_id)
        
        messages = [
            SystemMessage(content=instruction + schema + query ),
            HumanMessage(content=input)
        ]
        #You can modifify SystemMessage and HumanMessage. 
        #Normally people put instruction in system message and input in Humman message.

        #Have Fun 

        print(instruction + schema  +query)
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

        
        