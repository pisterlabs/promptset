from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from gpt4all import GPT4All
from langchain import PromptTemplate, LLMChain





from pdf2image import convert_from_path

loader = PyPDFLoader("docs/el3.pdf")
documents = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=64
)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(texts, embeddings, persist_directory="db")

print('Se han cargado los archivos en Chroma')

model_n_ctx = 1000
model_path = "./ggml-gpt4all-j-v1.3-groovy.bin"
llm = GPT4All(model='/home/alberto/atheneia/models/ggml-gpt4all-j-v1.3-groovy.bin',
              n_ctx=1000, backend="gptj", verbose=False)

print('Se ha cargado el LLM')
#llm = llms.GPT4All(
#    model="/home/alberto/atheneia/models/llama-2-7b-chat.ggmlv3.q2_K.bin",
#    # max_tokens=1000,
#    callbacks=callbacks,
#    verbose=True
#   )

# llm = llms.GPT4All(model="/home/alberto/atheneia/models/llama-2-7b-chat.ggmlv3.q2_K.bin")

template = """
You are a friendly chatbot assistant that responds in a conversational
manner to users questions. Keep the answers short, unless specifically
asked by the user to elaborate on something.

Question: {question}

Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# llm_chain = LLMChain(prompt=prompt, llm=llm)


qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
    return_source_documents=False,
    verbose=False
)

res = qa(f"""
    Como se calcula la autoinductancia
""")
print(res["result"])