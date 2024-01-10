"""
    TDS Experimental Controller.

    Description: Defines the basic rest endpoints for the TDS Module.
"""
import re
from logging import Logger

import openai
from fastapi import APIRouter, Depends
from sqlalchemy.engine.base import Engine

from tds.db.graph.neo4j import request_engine as request_graph_db
from tds.db.relational import request_engine as request_rdb
from tds.modules.provenance.utils import return_graph_validations
from tds.settings import settings

experimental_router = APIRouter()
logger = Logger(__name__)
valid_relations = return_graph_validations()

DB_DESC = "Valid relations include:\n"
# NOTE: Should we make these sentences more natural language related?
for relation, mapping in valid_relations.items():
    for dom, codom in mapping:
        DB_DESC += f"{dom}-[relation]->{codom}\n"

PREAMBLE = """
I will type "Question:" followed by a question or command in English like "Question: Count all Publications" and you will return a
single line print "Query:" Followed by an openCypher query like "Query: `match (p:Publication) return count(p)`.
"""

EXAMPLES = """
Question: Match all nodes in the database
Query: `Match (n) return n`

Question: Which composed models are derived from Paper with ID of 12?
Query: `Match (m:Model)-[r *1..]->(i:Intermediate)-[r2:EXTRACTED_FROM]->(p:Publication) where p.id=12 return m`

Question: Which datasets are associated with which model primitives?
Query: `match (d:Dataset)-[r *1..]-(i:Intermediate) return d,r,i`

Question: Return simulators that rely on Primitive with ID of 19 or 12
Query: `match (p:Plan)-[r *1..]->(i:Intermediate) where i.id=19 or i.id=12 return p,r,i`

Question: Which simulators were created by User with ID 999?
Query: `match (p:Plan)-[r:USES]->(mr:ModelRevision) where r.user_id=999 return p`

Question: What are prior versions of this composed model with id 5?
Query: `Match (rev:ModelRevision)<-[r:BEGINS_AT]-(m:Model) where (m.id=5) match (rev2:ModelRevision)<-[r2 *1.. ]-(rev) return rev,rev2`
"""


@experimental_router.get("/cql")
def convert_to_cypher(
    query: str,
) -> str:
    """
    Convert English to Cypher.
    """
    user_query = f"Question: {query}\nQuery: "
    prompt = PREAMBLE + DB_DESC + "\n" + EXAMPLES + "\n" + user_query
    completion = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=64,
        api_key=settings.OPENAI_KEY,
    )
    response = completion.choices[0].text
    matched_expression = re.compile(r"`([^`]*)`").fullmatch(response.strip())
    if matched_expression is None:
        raise Exception("OpenAI did not generate a valid response")
    cypher: str = matched_expression.groups(1)[0]
    logger.info("converted '%s' TO `%s`", query, cypher)
    return cypher


@experimental_router.get("/provenance")
def search_provenance(
    query: str,
    # rdb: Engine = Depends(request_rdb),
    # graph_db=Depends(request_graph_db),
) -> str:
    """
    Convert English to Cypher.
    """
    # cypher = convert_to_cypher(query)
    raise NotImplementedError


@experimental_router.get("/set_properties")
def set_properties(
    rdb: Engine = Depends(request_rdb),
    graph_db=Depends(request_graph_db),
) -> bool:
    """
    Modify DB contents to work with Neoviz
    """
    # Importing ProvenanceHandler here to bypass circular import issue.
    # pylint: disable-next=import-outside-toplevel
    from tds.db.graph.provenance_handler import ProvenanceHandler

    if settings.NEO4J_ENABLED:
        print("Neo4j is set")
        provenance_handler = ProvenanceHandler(rdb=rdb, graph_db=graph_db)
        success = provenance_handler.add_properties()
        return success
    return False
