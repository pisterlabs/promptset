# ctransformers == 0.2.24
# gradio
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings

loader = GenericLoader.from_filesystem(
    path="data",
    glob="**/*",
    suffixes=[".py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
)
documents = loader.load()
print(f"{documents=}")

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000,
    chunk_overlap=200
)
texts = python_splitter.split_documents(documents)
print(f"{texts=}")

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

print("embedding")
embedding_model_name = "BAAI/bge-base-en"
embedding_model_name = "models/bge-base-en"
embedding = HuggingFaceBgeEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

print("db...")
# db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
db = Chroma.from_documents(texts, embedding)
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8},
)

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain



# 'codellama-13b-instruct.Q4_K_M.gguf' 16G RAM
model_path = 'models/CodeLlama-34B-Instruct-GGUF/codellama-34b-instruct.Q4_K_M.gguf'

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
print("loading llm")
llm = LlamaCpp(
    model_path=model_path,
    n_ctx=5000,
    n_gpu_layers=40,
    n_threads=15,
    n_batch=512,
    f16_kv=True,
    callback_manager=callback_manager,
    verbose=True,
)
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
print("LLM loaded=========")

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

system_prompt = """You are a helpful assistant, you will use the provided context to answer user questions.
Read the given context before answering questions and think step by step. If you can not answer a user question based on
the provided context, inform the user. Do not use any other information for answering user"""

instruction = """
Context: {context}
User: {question}"""


def prompt_format(instruction=instruction, system_prompt=system_prompt):
    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template


template = prompt_format()
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

#Docs
question = "How can I use C# to initialize a Web Project?"
docs = retriever.get_relevant_documents(question)

print("Chain")
# Chain
chain = load_qa_chain(
    llm,
    chain_type="stuff",
    prompt=QA_CHAIN_PROMPT
)

print("Run")
# Run
result = chain({
    "input_documents": docs,
    "question": question
    },
    return_only_outputs=True
)
print(f"{result=}")
