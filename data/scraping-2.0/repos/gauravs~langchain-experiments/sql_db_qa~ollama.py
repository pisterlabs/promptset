import sys

from langchain.llms import Ollama
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

db = SQLDatabase.from_uri("sqlite:///db/Chinook.db")
llm = Ollama(model="phind-codellama", base_url="http://host.docker.internal:11434", temperature=0, verbose=True)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

def main():
  args_string = ' '.join(sys.argv[1:])
  db_chain.run(args_string)

if __name__ == "__main__":
  main()
