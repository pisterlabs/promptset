from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from pdf2image import convert_from_path

from pdf2image import convert_from_path

# Set the path to the Poppler binaries manually
poppler_path = r'C:\Users\HP\Desktop\intern-pyth\BERT\poppler-23.07.0\Library\bin'  # Adjust this path according to your installation

pdf_file = 'ms-financial-statement.pdf'
images = convert_from_path(pdf_file, poppler_path=poppler_path)
len(images)

# Continue with the code to extract text or use the images...

images[0]

images[1]

loader = PyPDFLoader("ms-financial-statement.pdf")

documents = loader.load_and_split()

len(documents)

print(documents[0].page_content)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
texts = text_splitter.split_documents(documents)
len(texts)
     
print(texts[0].page_content)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma.from_documents(texts, embeddings, persist_directory="db")

llm = GPT4All(model="./ggml-gpt4all-j-v1.3-groovy.bin", n_ctx=1000, backend="gptj", verbose=False)
model_n_ctx = 1000
# model_path = "./ggml-gpt4all-j-v1.3-groovy.bin"
# llm = GPT4All(model="./ggml-gpt4all-j-v1.3-groovy.bin", n_ctx=1000, backend="gptj", verbose=False)
    
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    verbose=False,
)
res = qa(f"""
     How much is the dividend per share during during 2022?
     Extract it from the text.
 """)
 print(res["result"])