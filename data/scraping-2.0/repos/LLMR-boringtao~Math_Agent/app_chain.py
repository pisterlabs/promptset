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

llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0.0, model=llm_model)

problem = """The Education Bureau sends a batch of books to schools. \
    If each school sent 8 boxes of Chinese language books and 6 boxes of maths books, then the remaining maths books are three times the Chinese language books. \
    If each school sent 9 boxes of Chinese language books and 8 boxes of maths books, then the shortage of Chinese language books is three times the maths books. \
    Assuming that the books are all in whole boxes, how many boxes did the Education Bureau originally have at least?
"""

role = """
    You are an experienced maths teacher. \
    You are good at solving descriptive maths problems by converting them into mathematical equations. \
    You always solve maths problems in a step-by-step way to be sure having the correct answer. \
    You use external tools such as Wolfram Alpha for any mathematical calculation. \
    Once you have the equation, you use external tools such as Wolfram Alpha to solve the equations.    
"""

method = """
    Take a deep breath. \
    Solve the problem by proposing as many different methods as possible and finding the most logically consistent one as the final answer. \
    Hints: ”algebra”;”optimization”,”equation”,”logical reasoning”
"""


prompt_solution = f"""
    As {role}, solve the problem that is delimited by triple backticks by using {method}. \
    problem: ```{problem}```
"""

prompt_one = ChatPromptTemplate.from_template(
    "What is the mathmatical equation of the {problem}?"
)

chain_one = LLMChain(llm=llm, prompt=prompt_one)

prompt_two = ChatPromptTemplate.from_template(
    "what is the solution of the :{equation}"
)

chain_two = LLMChain(llm=llm, prompt=prompt_two)

overall_chain = SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True)

overall_chain.run(problem)