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

def memory():
    conversational_memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    return conversational_memory


def question_factory(prof_info_1, conversational_memory):
    system_message = """
            "You are a Virtual Assistant to Plaksha University"
            "You should help use with question about a professor or question relating to the recommending a professor to on a course."
            "Pass in any question I passed to you  directly to the tools do not change anything"
            "Note all professors are always available"
            "You should only talk within the context of problem."
            "Pass the question directly to the tool"
            """
    tools = [
        Tool(
            name="qa_prof_1",
            func=prof_info_1.run,
            description="Use this tool always",
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
    # embed and store
    vectorstore = embed()

    # prompt
    QA_CHAIN_PROMPT = template_1()

    # setting up first retrieval
    qa_chain = retrieval(QA_CHAIN_PROMPT, vectorstore)

    # running the first input
    executor = question_factory(qa_chain, memory())

    # Loop
    while True:
        user_input = input("Input your question \n")
        answer = executor.run(user_input)
        print(answer)


if __name__ == "__main__":
    main()
