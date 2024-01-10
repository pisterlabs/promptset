from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate

from llm import llm
from graph import graph

# Cypher Prompt to define to only use schema provided
CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies.
Convert the user's question based on the schema.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Example Cypher Statement:
1. Who reviewed the movie?
```
MATCH (p:Person)-[r:REVIEWED]->(:Movie {{title: 'Movie title'}})
RETURN p.name as reviewer, r.rating as rating, r.summary as summary
```

Schema:
{schema}

Question:
{question}
"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

# Graph enabled semantic search to use the Neo4j grpah
cypher_qa = GraphCypherQAChain.from_llm(
    llm,         
    graph=graph,
    # verbose=True,
    cypher_prompt=cypher_prompt
)
