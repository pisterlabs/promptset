from langchain.agents import initialize_agent
from langchain.agents.types import AgentType
from langchain.tools import Tool
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from config import OPENAI_API_KEY
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# excel file loader function


def Loader(excel_file_url, document_loader):
    loader = document_loader(
        excel_file_url
    )
    data_to_split = loader.load()
    return data_to_split

# Splitters


def document_splitter(data_to_split, chunk_size, chunk_overlap, splitting_function):
    text_splitter = splitting_function(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_splits = text_splitter.split_documents(data_to_split)
    return all_splits


# embedding functions which uses Chroma vectore store
def embed_documents(splitted_document):
    vectorstore = Chroma.from_documents(
        documents=splitted_document, embedding=OpenAIEmbeddings())
    return vectorstore


def embed():
    persist_directory = "db"
    return Chroma(embedding_function=OpenAIEmbeddings(),
                  persist_directory=persist_directory)

# Template for the model one


def template_1():
    template = """You are ConnectToLearn owned by Plaksha University. You help with connecting user who have a particular interest in a topic to a Professor who have experience in such topic. You should give professors name and information about them based on the field of interest of the student in the Question asked by the student. You also can answer questions regarding a professor. When you are asked about a professor, ensure you provide a well detailed answer to it. 
    Ensure you go through the below context before answering a question, if you do not know the answer just tell the user that you do not have any information about that.
    You can only get the answer from the context:-
    {context}

    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=template,)
    return QA_CHAIN_PROMPT


def retrieval(QA_CHAIN_PROMPT, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectorstore.as_retriever(),
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    return qa_chain


# This function get the question and use relevant_documents to search some key words
def search_relevant_doc(vectorestore, question):
    retrieval = MultiQueryRetriever.from_llm(
        retriever=vectorestore.as_retriever(), llm=llm,
    )  # "lines" is the key (attribute name) of the parsed output
    # Results
    unique_docs = retrieval.get_relevant_documents(
        query=question
    )
    return (unique_docs)


def memory():
    conversational_memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    return conversational_memory


def question_factory(prof_info_1, prof_info_2, conversational_memory):
    system_message = """
            You are a Virtual Assistant to Plaksha University"
            "Always use all the tool provided"
            "You should help use with question about a professor or question relating to the recommending a professor to on a course."
            "pass in any question I passed to you  directly to the tools do not change anything"
            "You should only talk within the context of problem."
            "You should use the two tools for the answer"
            """
    tools = [
        Tool(
            name="qa_prof_1",
            func=prof_info_1.run,
            description="Use this tool regarding any recommedation question",
        ),
        Tool(
            name="qa-prof_2",
            func=prof_info_2.run,
            description="Use this tool for searching about a specific professor",
        )
    ]
    executor = initialize_agent(
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        memory=conversational_memory,
        agent_kwargs={"system_message": system_message},
        verbose=True,
    )
    return executor


def main():
    # Load documents
    data_to_split = Loader("../excel/faculty.xlsx", UnstructuredExcelLoader)

    # split documents
    all_splits = document_splitter(
        data_to_split, 1000, 200, RecursiveCharacterTextSplitter)

    # embed and store
    vectorstore = embed()

    # prompt
    QA_CHAIN_PROMPT = template_1()

    # setting up first retrieval
    qa_chain = retrieval(QA_CHAIN_PROMPT, vectorstore)

    while True:
        # Getting the first question from the user
        user_input = input("Input your question \n")

        # Creating the first model based on the user query
        prof = search_relevant_doc(vectorstore, user_input)
        prof_retrieval = embed_documents(prof)
        prof_info = retrieval(QA_CHAIN_PROMPT, prof_retrieval)
        # Loop
        executor = question_factory(qa_chain, prof_info, memory())

        # running the first input
        answer = executor.run(user_input)
        print(answer)


if __name__ == "__main__":
    main()
