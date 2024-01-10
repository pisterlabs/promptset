import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate

load_dotenv()

llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_KEY"))
graph = Neo4jGraph(url=os.getenv("NEO4J_URL"), username=os.getenv("NEO4J_USER"), password=os.getenv("NEO4J_PASSWORD"))

cypher_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=PromptTemplate(
        template="""
        You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
        Convert the user's question based on the schema.

        Schema: {schema}
        Question: {question}
        """,
        input_variables=["schema", "question"],
    ),
    verbose=True
)

user_question = "What role did Tom Hanks play in Toy Story?"
cypher_chain.run(user_question)