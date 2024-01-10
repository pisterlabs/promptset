from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import os
import tiktoken

os.environ["OPENAI_API_KEY"] = ''
dir = '/Users/haricharan/Documents/Projects/Reomnify/SSD-59568708'
pdfs = os.listdir(dir)

parsed_text = ""
for pdf in pdfs:
    pdf_path = os.path.join(dir, pdf)
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        parsed_text += page.extract_text()

text_chunks = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200).split_text(parsed_text)
knowledge_base = FAISS.from_texts(text_chunks, OpenAIEmbeddings(model='text-embedding-ada-002'))

while(True):
    query = input("Ask me anything: ")
    relavent_docs = knowledge_base.similarity_search(query)

    chain = load_qa_chain(llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0))
    response = chain.run(input_documents=relavent_docs, question=query)
    print(response)

