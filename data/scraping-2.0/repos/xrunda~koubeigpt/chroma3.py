import os

from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
import json


os.environ["OPENAI_API_KEY"]='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'

embeddings = OpenAIEmbeddings()
file_path = 'formatted_documents_split_columns_corrected_100.csv'
data = pd.read_csv(file_path)
docs=[]
for index, row in data.iterrows():
    page_content = row['page_content']
    metadata = row['metadata'].replace("'", '"')
    docs.append(Document(page_content=page_content,metadata=json.loads(metadata)))


# loader = CSVLoader("formatted_documents_split_columns_corrected_100.csv")
# documents = loader.load()

# text_splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0)
# loader_docs = text_splitter.split_documents(documents)

# print(docs[:2])
vectorstore = Chroma.from_documents(docs, embeddings)


from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="brand",
        description="汽车品牌",
        type="string",
    ),
    AttributeInfo(
        name="model",
        description="汽车型号",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="上市年份",
        type="string",
    ),
    AttributeInfo(
        name="price", 
        description="售价", 
        type="string"
    ),
    AttributeInfo(
        name="rating", 
        description="车型特点", 
        type="string"
    ),
]
document_content_description = "汽车评论"
llm = OpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True
)

# This example only specifies a relevant query
print(retriever.get_relevant_documents(query="丰田卡罗拉外观"))