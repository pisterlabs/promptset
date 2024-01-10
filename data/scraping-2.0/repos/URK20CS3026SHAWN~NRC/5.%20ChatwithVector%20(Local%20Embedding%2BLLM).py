from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import Chroma

n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
use_mlock = True  # Force system to keep model in RAM.
# Make sure the model path is correct for your system!
model_path="Models/vicuna-7b-v1.5.Q5_K_M.gguf"

embedding = LlamaCppEmbeddings(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    #f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    use_mlock = True)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding)

# question = "What are the approaches to Task Decomposition?"
# docs = vectorstore.similarity_search(question)
# len(docs)


from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=512,
    #f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
    use_mlock = True
)

user_question = "What are the approaches to Task Decomposition?"

# from langchain.chains.question_answering import load_qa_chain
# chain = load_qa_chain(llm, chain_type="stuff")
# response = chain.run(input_documents=, question=user_question)

from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT},
)

qa_chain({"query": user_question})