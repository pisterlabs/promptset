import json
import pickle
import os 
from langchain.chat_models import ChatOpenAI
from langchain.agents.tools import Tool


from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = API_KEY



course_info = None
with open("./data/courses.json", "r") as f:
        course_dict = json.load(f)
        course_info = course_dict["courses"]

def specific_course_info(course_num):
    res = []
    for course in course_info:
        if course["department"] == "CSCI" and course["number"] == int(course_num):
            res.append(course)
    
    return res if res else "Invalid course number"



cs_db = None
with open("./data/dev/db_cs_with_sources.pkl", "rb") as f:
    cs_db = pickle.load(f)

def query_cs_db(query):
    embedding_vector = embeddings.embed_query(query)
    return cs_db.similarity_search_by_vector(embedding_vector, k=6)
    # return cs_db.similarity_search(query, k=6)



from langchain.utilities import WikipediaAPIWrapper
wikipedia = WikipediaAPIWrapper()



tools = [
    Tool(
        name="CS major requirements and department info",
        func=query_cs_db,
        description="can be used to look up computer science major requirements, department info, course descriptions, and the program cirriculum for an input keyword or query. Useful for answering questions about major requirements, study away, advanced placement, and other FAQs about the CS department or what is taught in courses"
    ),
    Tool(
        name = "Specific CSCI course info",
        func=specific_course_info,
        description="use this to look up a information about a specific CSCI course. This should be used if the question mentions a particular CS course number. Note that the provided input *must* be just the three-digit, integer course number without any prefix or other text"
    ),
    Tool(
        name = "Wikipedia",
        func=wikipedia.run,
        description="use this when the user asks you to consult wikipedia about a certain topic, person, or term"
    )
]



from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm=ChatOpenAI(temperature=0)
agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=False, memory=memory)

q1 = "Hi, i am Mark"
print(q1)
print("RAGdoll: " + str(agent_chain.run(input=q1)))
print()

q2 = "What is taught in CSCI 374?"
print(q2)
print("RAGdoll: " + str(agent_chain.run(input=q2)))
print()

q3 = "Cool! What is reinforcement learning? Consult wikipedia and explain it to me in simple terms."
print(q3)
print("RAGdoll: " + str(agent_chain.run(input=q3)))