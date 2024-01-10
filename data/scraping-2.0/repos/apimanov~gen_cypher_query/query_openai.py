from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate

uri='http://127.0.0.1:5000/v1'
model = "facebook_opt-1.3b"

OPENAI_API_KEY='sk-111111111111111111111111111111111111111111'
llm = ChatOpenAI(base_url=uri, model=model, openai_api_key=OPENAI_API_KEY)

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="",
    password=""
)

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about infrastructure objects.
Convert the user's question based on the schema.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

If no data is returned, do not attempt to answer the question.
Only respond to questions that require you to construct a Cypher statement.
Do not include any explanations or apologies in your responses.

Examples: Here are a few examples of generated Cypher statements for particular questions:
# How many pods have default namespace?
MATCH (n:KubernetesNamespace {{name:'default'}})-[:HAS_POD]->(m) RETURN count(m) as pods
# How many containers have sample namespace?
MATCH (n:KubernetesNamespace {{ name: 'sample' }})-[:HAS_POD]->(pod)-[:HAS_CONTAINER]->(containers) return count(containers) as containers

Schema: {schema}
Question: {question}
"""

cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"],
)

cypher_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True,
    validate_cypher=True
)
print(cypher_generation_prompt)
out = cypher_chain.run("How many pods have default namespace?")
print (out)
