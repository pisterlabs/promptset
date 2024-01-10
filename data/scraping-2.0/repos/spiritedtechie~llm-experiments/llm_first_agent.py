from dotenv import load_dotenv
import os

load_dotenv(".env")

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI


# model_name = gpt-3.5-turbo, text-davinci-003
# llm = OpenAI(
#     model_name="text-davinci-003",
#     temperature=0,
#     openai_api_key=os.getenv("OPENAI_API_KEY"),
# )
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)
# llm = ChatOpenAI(model_name='gpt-4')

# deep lake dataset
storeDocuments = False  # change me
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
my_activeloop_org_id = os.getenv("ACTIVE_LOOP_ORG_ID")
my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

#
# Store some documents as embeddings in deep lake
#
if storeDocuments == True:
    texts = [
        "Napoleon Bonaparte was born in 15 August 1769",
        "Louis XIV was born in 5 September 1638",
    ]
    docs = text_splitter.create_documents(texts)
    db.add_documents(docs)


#
# Agent that uses the deeplake as a retrieval tool
#
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=db.as_retriever()
)

search = GoogleSearchAPIWrapper(
    google_api_key=os.getenv("GOOGLE_API_KEY"), google_cse_id=os.getenv("GOOGLE_CSE_ID")
)

prompt = PromptTemplate(
    input_variables=["query"], template="Write a summary of the following text: {query}"
)

summarize_chain = LLMChain(llm=llm, prompt=prompt)

tools = [
    Tool(
        name="Google search",
        description="useful for answering questions about current events.",
        func=search.run,
    ),
    Tool(
        name="Date of birth finder",
        func=retrieval_qa.run,
        description="useful for finding date of births.",
    ),
    Tool(
        name="Summarizer",
        func=summarize_chain.run,
        description="useful for summarizing texts to specific number of words",
    ),
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=6,
)

agent.verbose = True


with get_openai_callback() as cb:
	response = agent.run("When was Napoleon born?")
	print(response)
        
	response = agent("What's the latest news about the wildfires in Greece? Then summarise the result.")
	print(response)
	print(cb)