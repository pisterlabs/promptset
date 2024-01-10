from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings


import pandas as pd
import json
import os
from langchain.schema import Document


os.environ["OPENAI_API_KEY"]='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'

embeddings = OpenAIEmbeddings()
# embeddings = ModelScopeEmbeddings(model_id="xrunda/m3e-base",model_revision="v1.0.4")
file_path = 'modelY_transformed_corrected.csv'
data = pd.read_csv(file_path)
docs=[]
for index, row in data.iterrows():
    page_content = row['page_content']
    metadata = row['metadata'].replace("'", '"')
    docs.append(Document(page_content=page_content,metadata=json.loads(metadata)))
# vectorstore = Chroma.from_documents(docs, embeddings)
Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db_modelY")