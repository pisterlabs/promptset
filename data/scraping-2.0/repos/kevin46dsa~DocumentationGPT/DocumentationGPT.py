import os
import sys

import envVar
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

os.environ["OPENAI_API_KEY"] = envVar.APIKEY

#from langchain.chat_models import ChatOpenAI
#llm = ChatOpenAI(openai_api_key="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")

query = sys.argv[1]
print(query)

loader = TextLoader('Data.txt')

index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query))