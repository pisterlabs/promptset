# E. Culurciello, June 2023
# test langchain

# useful link for this:
# https://gist.github.com/scriptsandthings/75c38c54e05dd20d65fd83a9bd522406

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA


# llm model:
VICUNA = "./vicuna-7b-1.1.ggmlv3.q4_0.bin"

# loading documents:
loader = TextLoader("data.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

# llm for dialogue:
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path=VICUNA, 
    callback_manager=callback_manager,
    verbose=False,
)

embeddings = LlamaCppEmbeddings(model_path=VICUNA)
docsearch = Chroma.from_documents(texts, embeddings)

MIN_DOCS = 1
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": MIN_DOCS}),
    verbose=False,
)

query = """Identify the name of the black hole. Provide a concise answer."""
answer = qa.run(query)

print("\n\nAnswer:", answer)