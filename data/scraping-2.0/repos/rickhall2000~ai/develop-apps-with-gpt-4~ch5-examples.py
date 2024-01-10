from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain, OpenAI, ConversationChain
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.documernt_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

def prompt_demo():
    template = """Question {question}
    Let's think step by step.
    Answer: """

    propmt = PromptTemplate(template=template, input_variables=["question"])
    llm = ChatOpenAI(model_name="gpt-4")
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    question = """ What is the population of the capital of the country where the
    Olympic Games were held in 2016"""
    llm_chain.run(question)

    
def agent_demo():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    tools = load_tools(["wikipedia", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    question = """What is the square root of the population of the capital of the country 
    where the Olympic Games were held in 2016?"""
    agent.run(question)

    
def chatbot_demo():
    chatbot_llm = OpenAI(model_name='text-ada-001')    
    chatbot = ConversationChain(llm=chatbot_llm, verbose=True)
    chatbot.predict(input="Hello")

def loader_example():
    loader = PyPDFLoader("ExplorersGuide.pdf")
    pages = loader.load_and_split()
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(pages, embeddings)
    q = "What is Link's traditional outfit color?"
    db.similarity_search(q)[0]
    llm = OpenAI()
    chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever())
    chain(q, return_only_outputs=True)
        

    