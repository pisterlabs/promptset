from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler,
)  # for streaming response
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma

from constants import PERSIST_DIRECTORY
from model_loader import load_model

load_dotenv()

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm_model = load_model(
    model_id="TheBloke/Llama-2-7b-Chat-GGUF",
    model_basename="llama-2-7b-chat.Q4_K_M.gguf",
    loader_type="hf_local",
)

system_prompt = """
You are a helpful GuardMe Claim Examiner Assistance.
You need to learn all the content in the context.
You should help GuardMe customers to know any information about us from the context.
You should instruct the customers how they can submit their claims.
You should instruct the customers how they can check their claims status.
You should take down the customers' information and submit their claims.
You should take decision about the customers' claims, based on the information you have.
You will use the provided context to answer user questions."""
# system_prompt = """You are a helpful assistant, you will use the provided context to answer user questions.
# Read the given context before answering questions and think step by step. If you can not answer a user question based on
# the provided context, inform the user. Do not use any other information for answering user. Provide a detailed answer to the question."""

instruction = """
            Context: {history} \n {context}
            User: {question}"""

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST


prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=prompt_template,
)


embeddings = HuggingFaceEmbeddings()
db = Chroma(
    embedding_function=embeddings,
    persist_directory=PERSIST_DIRECTORY,
)

retriever = db.as_retriever()


memory = ConversationBufferMemory(input_key="question", memory_key="history")

chat = RetrievalQA.from_chain_type(
    llm=llm_model,
    chain_type="stuff",
    verbose=True,
    retriever=retriever,
    # return_source_documents=True,
    callbacks=callback_manager,
    # chain_type_kwargs={"prompt": prompt},
    chain_type_kwargs={"prompt": prompt, "memory": memory},
)
