from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
#--------------------
# Clone
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import LlamaCpp
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers import MMR
#--------------------
# from git import Repo
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

#--------------------
repo_path = "/Users/rlm/Desktop/test_repo"
# repo = Repo.clone_from("https://github.com/hwchase17/langchain", to_path=repo_path)

#--------------------
# Load
loader = GenericLoader.from_filesystem(
    repo_path+"/libs/langchain/langchain",
    glob="**/*",
    suffixes=[".py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
)
documents = loader.load()
len(documents)

#--------------------
python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, 
                                                               chunk_size=2000, 
                                                               chunk_overlap=200)
texts = python_splitter.split_documents(documents)
len(texts)

#--------------------
db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
retriever = MMR(db, search_kwargs={"k": 8})

#--------------------
llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], model="text-davinci-002") 
memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

#--------------------
question = "How can I initialize a ReAct agent?"
result = qa(question)
result['answer']

#--------------------
questions = [
    "What is the class hierarchy?",
    "What classes are derived from the Chain class?",
    "What one improvement do you propose in code in relation to the class herarchy for the Chain class?",
]

for question in questions:
    result = qa(question)
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")

#--------------------
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="/Users/rlm/Desktop/Code/llama/code-llama/codellama-13b-instruct.Q4_K_M.gguf",
    n_ctx=5000,
    n_gpu_layers=1,
    n_batch=512,
    f16_kv=True,
    callback_manager=callback_manager,
    verbose=True,
    client=None,
    n_parts=None,
    seed=None,
    logits_all=None,
    vocab_only=None,
    use_mlock=None,
    n_threads=None,
    suffix=None,
    logprobs=None
)

#--------------------
llm("Question: In bash, how do I list all the text files in the current directory that have been modified in the last month? Answer:")

#--------------------
# Prompt
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# Chain
chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_CHAIN_PROMPT)

# Docs
question = "How can I initialize a ReAct agent?"
docs = retriever.get_relevant_documents(question)

# Run
chain({"input_documents": docs, "question": question}, return_only_outputs=True)


#--------------------
from langchain import hub

QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-default")

#--------------------
# Docs
question = "How can I initialize a ReAct agent?"
docs = retriever.get_relevant_documents(question)

# Chain
chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_CHAIN_PROMPT)

# Run
chain({"input_documents": docs, "question": question}, return_only_outputs=True)

#--------------------
