from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import (RecursiveCharacterTextSplitter,Language)

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1250, separators=["\n\n", "\n","as","assert","async","await","break","class","continue","def","del","elif","else","except","exec","finally","for","from","global","if","import","lambda","nonlocal","pass","print","raise","return","try","while","with","yield","match","case"," ",""],
                                             chunk_overlap=0)
texts = text_splitter.split_text("state_of_the_union")
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=4000, chunk_overlap=0
)
