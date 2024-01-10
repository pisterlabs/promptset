from pydantic import BaseModel, Field
from typing import List
import os
import chainlit as cl

class Node(BaseModel):
    id: int
    label: str
    color: str

class Edge(BaseModel):
    source: int
    target: int
    label: str
    color: str = "black"

class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(..., default_factory=list)
    edges: List[Edge] = Field(..., default_factory=list)
    
from openai import OpenAI
import instructor
api_key = os.environ.get("OPENAI_API_KEY")
# Adds response_model to ChatCompletion
# Allows the return of Pydantic model rather than raw JSON

client = instructor.patch(OpenAI(api_key=api_key))

def generate_graph(input) -> KnowledgeGraph:
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Help me understand the following by describing it as a detailed knowledge graph: {input}",
            }
        ],
        response_model=KnowledgeGraph,
    )  # type: ignore
    
from graphviz import Digraph

def visualize_knowledge_graph(kg: KnowledgeGraph):
    dot = Digraph(comment="Knowledge Graph")

    # Add nodes
    for node in kg.nodes:
        dot.node(str(node.id), node.label, color=node.color)

    # Add edges
    for edge in kg.edges:
        dot.edge(str(edge.source), str(edge.target), label=edge.label, color=edge.color)

    # Render the graph
    dot.render("knowledge_graph.gv", view=True)
    
    
graph: KnowledgeGraph = generate_graph("Teach me about paging and other related os topics.")
visualize_knowledge_graph(graph)