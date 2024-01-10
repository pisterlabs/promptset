from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import json
import ast
import click
load_dotenv(verbose=True, override=True)
del load_dotenv


GRAPH = Neo4jGraph(
    url="bolt://localhost:7687", 
    username="neo4j", 
    password="pleaseletmein"
)


SYNTH_PROMPT_TEMPLATE = """\
You are a chef. You are great at describing dishes. You are so good because you are able to break down dishes into their component parts, \
describe the component parts, and then put them together to describe the dish as a whole. \
You are also good at describing the cooking process and the taste of each ingredient. \
You are also good at describing the overall flavor of the dish. \

Here is a dish:
{input}"""


GRAPH_FEATURE_EXTRACTION_TEMPLATE = """\
You are a graph database architect. You are great at extracting relational information from text. \
You are so good because you are able to break down text into its component parts, \
extract the relational information from the component parts, and then put them together to extract relational information from the text as a whole. \

Here is some text:
{input}"""


NEO4J_GRAPH_DB_QUERY_MAKER_TEMPLATE = """\
Could you please save the following information to the graph database? \

```raw_text
{input}
```

```preprocessed_text
{preprocessed_text}
```
"""


@click.command()
@click.option("--input_dish", '-i', default="A dish made with chicken, rice, and vegetables.", help="Dish to describe")
def main(input_dish: str):
    neo4j_graph_chain = GraphCypherQAChain.from_llm(ChatOpenAI(temperature=0.25), graph=GRAPH, verbose=True)
    GRAPH.refresh_schema()
    graph_db_schema = GRAPH.get_schema

    synth_prompt = SYNTH_PROMPT_TEMPLATE.format(input=input_dish)
    synth_llm = OpenAI(temperature=0.7, verbose=True)
    synth_result = synth_llm.predict(synth_prompt)
    print(f'[SYNTHESIZED RESULT]\n{synth_result}\n')
    
    feature_extraction_prompt = GRAPH_FEATURE_EXTRACTION_TEMPLATE.format(input=synth_result)
    feature_extraction_llm = OpenAI(temperature=0.7, verbose=True)
    feature_extraction_result = feature_extraction_llm.predict(feature_extraction_prompt)
    print(f'[FEATURE EXTRACTION RESULT]\n{feature_extraction_result}\n')

    query_maker_prompt = NEO4J_GRAPH_DB_QUERY_MAKER_TEMPLATE.format(
        input=synth_result, preprocessed_text=feature_extraction_result
    )
    query_maker_result = neo4j_graph_chain.run(query_maker_prompt)
    print(f'[GRAPH DB QUERY MAKER]\n{query_maker_result}\n')


if __name__ == "__main__":
    main()