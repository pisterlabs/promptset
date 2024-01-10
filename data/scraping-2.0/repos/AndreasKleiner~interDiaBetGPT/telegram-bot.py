import os
import sqlalchemy as db
from sqlalchemy import select
from sqlalchemy.sql import text
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate
import logging
import time
from telegram import Update, constants
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

# OpenAI part
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

# live database
db = SQLDatabase.from_uri("CONNECTION_URI", include_tables=['TABLE'])

llm = OpenAI(temperature=0)

_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. Make sure that the SQL query does not contain a semicolon!
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:

{table_info}

If someone asks for anything, they really always refer to the validateDataDiabetes3 table.

Every row in the table represents one person.

The likelihood of each person to get diabetes is in the column Prediccion. Keep that in mind when someone asks about likelihoods or probabilities of a person or people getting diabetes. 

Question: {input}"""
PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
)

db_chain = SQLDatabaseChain.from_llm(llm, db, prompt=PROMPT, verbose=True)

# Telegram part
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def message_received(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    
    if update.effective_chat.type is constants.ChatType.PRIVATE:
        logging.info("Query: " + text)        
        await context.bot.send_message(chat_id=update.effective_chat.id, text=db_chain(text)['result'])
        

if __name__ == '__main__':
    application = ApplicationBuilder().token('TELEGRAM_BOT_TOKEN').build()

    query_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), message_received)
    application.add_handler(query_handler)

    application.run_polling()
