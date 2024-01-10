"""Module providing a open AI function to retrieve knowledge graph."""
# Adapted version from
# https://github.com/tomasonjo/blogs/blob/master/llm/openaifunction_constructing_graph.ipynb?ref=blog.langchain.dev

from typing import List, Optional
from langchain.pydantic_v1 import Field, BaseModel
from langchain.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship
)


class Property(BaseModel):
    """A single property consisting of key and value"""

    key: str = Field(..., description="key")
    value: str | None = Field(..., description="value")


class Node(BaseNode):
    """A single node of Knowledge graph"""
    properties: Optional[List[Property]] = Field(
        None, description="List of node properties"
    )


class Relationship(BaseRelationship):
    """A relation between two nodes of a Knowledge graph"""
    properties: Optional[List[Property]] = Field(
        None, description="List of relationship properties"
    )


class KnowledgeGraph(BaseModel):
    """Generate a knowledge graph with entities and relationships."""

    nodes: List[Node] = Field(...,
                              description="List of nodes in the knowledge graph")
    rels: List[Relationship] = Field(
        ..., description="List of relationships in the knowledge graph"
    )
