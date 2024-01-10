# Adapted version from
# https://github.com/tomasonjo/blogs/blob/master/llm/openaifunction_constructing_graph.ipynb?ref=blog.langchain.dev

"""Retrieve knowledge graph from provided document"""
from typing import List, Dict, Any, Optional
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
    GraphDocument
)
from functions import (Node, Relationship, KnowledgeGraph)
from openai_prompts import (ner_prompt)

# This need to be coming from environment
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)


def format_property_key(s: str) -> str:
    """Format property key to be in camelCase"""
    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)


def props_to_dict(props) -> dict:
    """Convert properties to a dictionary."""
    properties = {}
    if not props:
        return properties
    for p in props:
        properties[format_property_key(p.key)] = p.value
    return properties


def map_to_base_node(node: Node) -> BaseNode:
    """Map the KnowledgeGraph Node to the base Node."""
    properties = props_to_dict(node.properties) if node.properties else {}

    # Add name property for better Cypher statement generation
    properties["name"] = node.id.title()
    return BaseNode(
        id=node.id.title(), type=node.type.capitalize(), properties=properties
    )


def map_to_base_relationship(rel: Relationship) -> BaseRelationship:
    """Map the KnowledgeGraph Relationship to the base Relationship."""
    source = map_to_base_node(rel.source)
    target = map_to_base_node(rel.target)
    properties = props_to_dict(rel.properties) if rel.properties else {}
    return BaseRelationship(
        source=source, target=target, type=rel.type, properties=properties
    )


def get_extraction_chain(allowed_nodes: Optional[List[str]] = None,
                         allowed_rels: Optional[List[str]] = None,
                         restricted_nodes: Optional[List[str]] = None,
                         existing_entities: Optional[List[str]] = None,):
    """OpenAI Langchain chain to retrieve NER using """
    # print(system_prompt)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                ner_prompt(allowed_nodes,
                           allowed_rels,
                           restricted_nodes,
                           existing_entities),
            ),
            (
                "human",
                "Use the given format to extract information from the following input: {input}",
            ),
            ("human", "Tip: Make sure to answer in the correct format"),
        ]
    )
    return create_structured_output_chain(KnowledgeGraph, llm, prompt, verbose=False)


def extract_graph(document: Document,
                  nodes: Optional[List[str]] = None,
                  rels: Optional[List[str]] = None,
                  restricted_nodes: Optional[List[str]] = None,
                  existing_entities: Optional[List[str]] = None,) -> GraphDocument:
    """Extract Knowledge Graph from a given document"""
    extract_chain = get_extraction_chain(
        nodes, rels, restricted_nodes, existing_entities)
    data = extract_chain.run(document.page_content)

    graph_document = GraphDocument(
        nodes=[map_to_base_node(node) for node in data.nodes],
        relationships=[map_to_base_relationship(rel) for rel in data.rels],
        source=document,
    )
    return graph_document
