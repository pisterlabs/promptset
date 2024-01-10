from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import JSONLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings

from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI
from langchain.agents import tool
from langchain.memory import ConversationSummaryBufferMemory

llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0.0, model=llm_model)
tools = load_tools(["llm-math"], llm=llm)

problem = """
    Classes A and B start from the school at the same time to go to a park that is 51 km away. Class A walks at a speed of 4 km/h, and Class B walks at a speed of 5 km/h. The school has a car that can travel at 60 km/h and can carry one class of students at a time. To ensure that both classes arrive at the park in the shortest time possible, how many kilometres do Classes A and B need to walk?
"""

role = """
    You are an experienced maths teacher. \
    You are good at solving descriptive maths problems by converting them into mathematical equations. \
    You always solve maths problems in a step-by-step way to be sure having the correct answer.     
"""

# APE (Automatic Prompt Engineer)
method = """
    Take a deep breath. \
    Follow the two examples below.\
    Find the equations by proposing as many different methods as possible and finding the most logically consistent one as the final answer. \
    Keep trying if you cannot find the equation. \
    Only output the equations and skip the analytical process. \
    Hints: ”algebra”;”optimization”,”equation”,”logical reasoning”

    Example 1:
    ###Problem:
    Classes A and B start from the school at the same time to go to a park that is 51 km away. Class A walks at a speed of 4 km/h, and Class B walks at a speed of 5 km/h. The school has a car that can travel at 60 km/h and can carry one class of students at a time. To ensure that both classes arrive at the park in the shortest time possible, how many kilometres do Classes A and B need to walk?
    ###Answer:
    1. Let Class A walk X km. Then the time Class A takes to walk is X/4 hours. In this time, the car travels 15X km.
    2. The car takes Class B to the point where they should start walking and then returns to meet Class A. The distance the car travels to take Class B is (15X+X)=8X. Therefore, Class B walks 51−8X km, and the time Class B takes to walk is 51−8X hours. 
    3. In this time, the car travels (51−8X)×12 km. At this point, the car needs to pick up Class A and return to the park.
    ###Solution:
    Equation: (51−8X)×12+(51−8X) +X=51.

    Example 2:
    ###Problem:
    A company organises a social event. 50% of the sales department participates, and 60% of the IT department participates. The number of people from the IT department who participate is 6 more than the number from the sales department. Later, due to urgent work, 60% of the remaining people in the sales department and 50% of the remaining people in the IT department are selected to form a 30-person work team. How many employees does the company have?
    ###Answer:
    1. Let the sales department have X people and the IT department have Y people. Then the number of people participating from the sales department is 0.5X and from the IT department is 0.6Y. The IT department has 6 more people participating than the sales department, so 0.6Y−0.5X=6.
    2. The number of people participating from the sales department is 0.5X, so the remaining number is also 0.5X. 60% of the remaining people are selected, which is 0.3X. The number of people participating from the IT department is 0.6Y, so the remaining number is 0.4Y. 50% of the remaining people are selected, which is 0.2Y. Therefore, 0.3X+0.2Y=30.
    ###Solution:
    Equation: 0.6Y−0.5X=6 and 0.3X+0.2Y=30.
"""

prompt = f"""as {role}, solve the problem that is delimited by triple backticks by using {method}.
    ###Problem: ```{problem}```
"""

prompt_template = ChatPromptTemplate.from_template(prompt)
prompt_solution = prompt_template.format_messages(
    problem=problem,
    role=role,
    method=method
)

response = llm(prompt_solution)
equation = response.content

agent= initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)

solution = agent(equation)
solution_process = solution['input']

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100, human_prefix="Student: ", ai_prefix="Teacher:")
memory.save_context({"input": "Hello, I am a maths student"}, {"output": "Hello, I am your maths teacher"})
memory.save_context({"input": "I am having trouble with this question"}, {"output": "What is the question?"})
memory.save_context({"input": f"{problem}"}, {"output": f"{solution_process}"})
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)

chat = conversation.predict(input="Can you explain how you get the solution again in fewer words?")
print(chat)