import psycopg2
import os
from langchain.llms import OpenAI
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Connect to the PostgreSQL database

## HERE YOU HANDLE LLM Parameter.

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    host="db-postgresql-nyc1-26343-do-user-9263663-0.b.db.ondigitalocean.com",
    port="25060",
    database="chatdata",
    user="chatdata",
    password="AVNS_AHdFDceQspT81IXGQ0w"
)


class TransactionChain():

    def __init__(self,tx_id:int) -> None:
        super().__init__()
        self.tx_id = tx_id

    def process_input(self, input:str, param = None) -> str:
        
        txc_id = self.create_transaction_chain(self.tx_id)
        self.save_input(txc_id,input)
        os.environ["OPENAI_API_BASE"] = "http://localhost:8010/workload_handler"
        os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # can be anything
        llm = OpenAI(streaming=False, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)
        output = llm(input)

        self.save_output(txc_id,output)
        #print(output)
        return output
        

    def create_transaction_chain(self, tx_id, txc_name = 'NA'):
        cur = conn.cursor()
        # Insert a new transaction into the transaction table
        cur.execute("INSERT INTO transaction_chain (tx_id, timestamp, agent_name) VALUES (%s, NOW(), %s) RETURNING txc_id", (tx_id, txc_name))
        new_txc = cur.fetchone()
        conn.commit()
        # Close the cursor
        cur.close()
        # Return the new transaction ID
        return new_txc[0]
    
    def save_input(self, txc_id, input_value):
        cur = conn.cursor()
        # Insert a new input transaction into the input_transaction table
        cur.execute("INSERT INTO input_transaction_chain (txc_id, input_value) VALUES (%s, %s)", (txc_id, input_value))
        conn.commit()
        # Close the cursor
        cur.close()

    def save_output(self, txc_id, input_value):
        cur = conn.cursor()
        # Insert a new input transaction into the input_transaction table
        cur.execute("INSERT INTO output_transaction_chain (txc_id, output_value) VALUES (%s, %s)", (txc_id, input_value))
        conn.commit()
        # Close the cursor
        cur.close()
        
    
