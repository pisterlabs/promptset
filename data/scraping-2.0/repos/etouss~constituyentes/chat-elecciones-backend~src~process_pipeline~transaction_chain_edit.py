import os
import openai
from sqlalchemy.sql import text
from dotenv import load_dotenv
from sqlalchemy.orm import Session


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

class TransactionChainEdit():

    def __init__(self, tx_id: int, db: Session) -> None:
        super().__init__()
        self.tx_id = tx_id
        self.db = db

    def process_input(self, input, instruction, param=None) -> str:
        
        txc_id = self.create_transaction_chain(self.tx_id)
        self.save_input(txc_id,'Input:'+input+'Instruction:'+instruction)

        response = openai.Edit.create(
            model='text-davinci-edit-001',
            input=input,
            instruction = instruction,
            temperature = 0
        )
        output = response.choices[0].text
        self.save_output(txc_id,output)
        return output

    def create_transaction_chain(self, tx_id, txc_name = 'NA'):
        # Insert a new transaction into the transaction table
        result_proxy = self.db.execute(
            text("INSERT INTO transaction_chain (tx_id, timestamp, agent_name) VALUES (:tx_id, NOW(), :txc_name) RETURNING txc_id"), 
            {
                "tx_id": tx_id,
                "txc_name": txc_name
            }
        )
        new_txc = result_proxy.fetchone()
        self.db.commit()
        # Return the new transaction ID
        return new_txc[0]
    
    def save_input(self, txc_id, input_value):
        # Insert a new input transaction into the input_transaction table
        self.db.execute(
            text("INSERT INTO input_transaction_chain (txc_id, input_value) VALUES (:txc_id, :input_value)"), 
            {
                "txc_id": txc_id,
                "input_value": input_value
            }
        )
        self.db.commit()

    def save_output(self, txc_id, input_value):
        # Insert a new input transaction into the input_transaction table
        self.db.execute(
            text("INSERT INTO output_transaction_chain (txc_id, output_value) VALUES (:txc_id, :input_value)"), 
            {
                "txc_id": txc_id,
                "input_value": input_value
            }
        )
        self.db.commit()
