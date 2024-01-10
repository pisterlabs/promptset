"""Neo4J Graph Database Repository"""
import json
import pathlib
from typing import List, Dict, Any
from langchain.graphs import Neo4jGraph
from langchain.graphs.graph_document import GraphDocument

# This need to be coming from environment
graph_db = Neo4jGraph()
graph_db.refresh_schema()


def save_graph_json(file_name: str, graph: GraphDocument):
    """Save Graph Data to disk"""
    file_path = pathlib.Path(file_name)
    with open(file_path, "w", encoding="utf-8") as swr:
        json.dump(graph.to_json(), swr, ensure_ascii=False)


def save_graph(document: GraphDocument):
    """Store finalized graph into database"""
    graph_db.add_graph_documents([document])


def execute_query(query: str, params: dict = {}) -> List[Dict[str, Any]]:
    """Run Query against DB"""
    return graph_db.query(query=query, params=params)
