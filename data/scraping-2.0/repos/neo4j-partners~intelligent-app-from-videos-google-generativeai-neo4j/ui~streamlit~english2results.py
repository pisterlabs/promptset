from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import VertexAI
from retry import retry
from timeit import default_timer as timer
import streamlit as st

host = st.secrets["NEO4J_HOST"]+":"+st.secrets["NEO4J_PORT"]
user = st.secrets["NEO4J_USER"]
password = st.secrets["NEO4J_PASSWORD"]
db = st.secrets["NEO4J_DB"]

codey_model_name = st.secrets["TUNED_CYPHER_MODEL"]
if codey_model_name == '':
    codey_model_name = 'code-bison'
    

CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j Cypher translator who understands the question in english and convert to Cypher strictly based on the Neo4j Schema provided and following the instructions below:
1. Generate Cypher query compatible ONLY for Neo4j Version 5
2. Do not use EXISTS, SIZE keywords in the cypher. Use alias when using the WITH keyword
3. Use only Nodes and relationships mentioned in the schema
4. Always enclose the Cypher output inside 3 backticks
5. Always do a case-insensitive and fuzzy search for any properties related search. Eg: to search for a Team name use `toLower(t.name) contains 'neo4j'`
6. Always use aliases to refer the node in the query
7. Cypher is NOT SQL. So, do not mix and match the syntaxes
Schema:
{schema}
Samples:
Question: What are the predictions about the Swans?
Answer: MATCH (e:Episode)-[:HAS_PREDICTION]->(p:Prediction) WHERE toLower(p.name) CONTAINS 'swans' RETURN p.name
Question: Who are the players mentioned in episode 1?
Answer: MATCH (e:Episode)-[:DISCUSSES_PLAYER]->(p:Player) WHERE e.episode = '1' RETURN p.name
Question: What are the top 5 common themes across all episodes combined?
Answer: MATCH (e:Episode)-[:HAS_THEME]->(t:Theme) RETURN t.name as theme, count(*) as num_themes ORDER BY num_themes DESC LIMIT 5
Question: Who are the most commonly talked coaches?
Answer: MATCH (e:Episode)-[:DISCUSSES_COACH]->(p:Coach) RETURN DISTINCT p.name as coach, count(e) as num_mentions  ORDER BY num_mentions DESC LIMIT 5
Question: What is the gist of episode 4?
Answer: MATCH (e:Episode) WHERE e.episode = '4' RETURN e.synopsis
Question: Which episodes do you recommend if I am a fan of the Bombers?
Answer: Match(e:Episode)-[:DISCUSSES_TEAM]->(t:Team) WHERE toLower(t.name) contains 'bombers' return e
Question: I follow Mason Cox. Which episodes do you recommend?
Answer: MATCH (e:Episode)-[:DISCUSSES_PLAYER]->(p:Player) WHERE toLower(p.name) CONTAINS 'mason cox' RETURN e

Question: {question}
Answer:"""
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

@retry(tries=5, delay=5)
def get_results(messages):
    start = timer()
    try:
        graph = Neo4jGraph(
            url=host, 
            username=user, 
            password=password
        )
        chain = GraphCypherQAChain.from_llm(
            VertexAI(
                    model_name=codey_model_name,
                    max_output_tokens=2048,
                    temperature=0,
                    top_p=0.95,
                    top_k=0.40), 
            graph=graph, verbose=True,
            return_intermediate_steps=True,
            cypher_prompt=CYPHER_GENERATION_PROMPT
        )
        if messages:
            question = messages.pop()
        else: 
            question = 'How many cases are there?'
        return chain(question)
    # except Exception as ex:
    #     print(ex)
    #     return "LLM Quota Exceeded. Please try again"
    finally:
        print('Cypher Generation Time : {}'.format(timer() - start))


