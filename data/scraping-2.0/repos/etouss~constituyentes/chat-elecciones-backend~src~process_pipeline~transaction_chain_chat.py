import os
import openai
from sqlalchemy.sql import text
from dotenv import load_dotenv
from sqlalchemy.orm import Session
import tiktoken


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def get_openai_substring(input_string, n):
    # Get the GPT-3 tokenizer
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # Tokenize the input string
    tokens = tokenizer.encode(input_string)

    # Return the substring with at most N tokens
    return tokenizer.decode(tokens[:n])

class TransactionChainChat():

    def __init__(self, tx_id: int, db: Session) -> None:
        super().__init__()
        self.tx_id = tx_id
        self.db = db

    def process_input(self, messages, param=None) -> str:
        
        txc_id = self.create_transaction_chain(self.tx_id)
        input = str(messages)
        self.save_input(txc_id,input)
        # Example OpenAI Python library request
        MODEL = "gpt-3.5-turbo"

        messages[0]['content'] = get_openai_substring(messages[0]['content'], 3800)

        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=messages,
            temperature=0
        )
                
        output = response.choices[0].message.content
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
