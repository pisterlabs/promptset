import sqlite3

import dotenv
from langchain import agents, llms, sql_database
from langchain.agents import agent_toolkits
import pandas as pd
import sqlalchemy


dotenv.load_dotenv()

# Utility block to extract content from a CSV file and load it into a SQLite table.
conn = sqlite3.connect("insurance.db")
df = pd.read_csv("datasets/insurance.csv")
df.to_sql("insurance", conn, if_exists="replace", index=False)
conn.close()

db_engine = sqlalchemy.create_engine("sqlite:///insurance.db")
# Instantiate the OpenAI model.
# Pass the "temperature" parameter which controls the RANDOMNESS of the model's output.
# A lower temperature will result in more predictable output, while a higher temperature
# will result in more random output. The temperature parameter is set between 0 and 1,
# with 0 being the most predictable and 1 being the most random.
llm = llms.OpenAI(temperature=0)
toolkit = agent_toolkits.SQLDatabaseToolkit(
    db=sql_database.SQLDatabase(db_engine), llm=llm
)

question = (
    "What is the highest charge?"
    " What is the average charge?"
    " What is the group that spends more, male of female?"
)
print(f"\nYOUR QUESTION IS:\n{question}")

executor = agents.create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type=agents.AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
result = executor.run(question)
print(f"\nTHE ANSWER TO YOUR QUESTION IS:\n{result}\n")
