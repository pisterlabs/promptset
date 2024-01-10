from research_copilot.data_loading.ca_code import (
    clean_extra_whitespace,
    embedding_index_name,
)
from unstructured.partition.html import partition_html
from langchain.schema import Document
from langchain.graphs.graph_document import GraphDocument, Node, Relationship
from research_copilot.db.graph import graph


def load_privacy_policy(url):
    elements = partition_html(url=url)
    nodes = []
    relationships = []
    source = Document(page_content="", metadata={"url": url})
    for e in elements:
        node = Node(
            id=e.to_dict()["element_id"],
            type="PrivacyPolicySection",
            properties={
                "text": clean_extra_whitespace(e.to_dict()["text"]),
                "url": url,
            },
        )
        if len(nodes) > 0:
            relationships.append(
                Relationship(source=nodes[-1], target=node, type="FOLLOWED_BY")
            )
        nodes.append(node)
    graph_doc = GraphDocument(nodes=nodes, relationships=relationships, source=source)
    graph.add_graph_documents([graph_doc], include_source=False)


def get_all_privacy_policy_sections():
    q = """
    MATCH (section:PrivacyPolicySection)
    OPTIONAL MATCH (section)-[r:HAS_HYPOTHETICAL_REQUIREMENT]->(h_req:PrivacyPolicyHypotheticalRequirement)
    RETURN section as node, h_req as hypothetical_requirements
    """
    results = graph.query(q)
    return results


def get_all_hypothetical_requirements():
    q = """
    MATCH (section:PrivacyPolicySection)-[r:HAS_HYPOTHETICAL_REQUIREMENT]->(h_req:PrivacyPolicyHypotheticalRequirement)
    RETURN section as node, h_req as hypothetical_requirements
    """
    results = graph.query(q)
    return results


def update_privacy_policy_section_properties(node_id: str, props: dict):
    q = f"""
    MATCH (n:PrivacyPolicySection)
    WHERE n.id = $node_id
    CALL apoc.create.setProperties(n, $prop_keys, $prop_values) YIELD node
    RETURN NULL
    """
    graph.query(
        q,
        params={
            "node_id": node_id,
            "prop_keys": list(props.keys()),
            "prop_values": list(props.values()),
        },
    )


# def create_hypothetical_requirements_nodes_for_privacy_policy_section(
#     section: dict, hypothetical_requirements_vectors: list
# ):
#     section_node = Node(
#         id=section["node"]["id"],
#     )
#     # TODO parametrize embedding name
#     embedding_name = "gte-large"
#     nodes = []
#     relationships = []
#     for i, (h_req, h_vec) in enumerate(hypothetical_requirements_vectors):
#         nodes.append(
#             Node(
#                 id=f"{section_node.id}-hreq-{i}",
#                 type="PrivacyPolicyHypotheticalRequirement",
#                 properties={"text": h_req, f"{embedding_name}": h_vec},
#             )
#         )
#         relationships.append(
#             Relationship(
#                 source=section_node,
#                 target=nodes[-1],
#                 type="HAS_HYPOTHETICAL_REQUIREMENT",
#             )
#         )

#     graph.add_graph_documents(
#         [
#             GraphDocument(
#                 nodes=nodes,
#                 relationships=relationships,
#                 source=Document(page_content="", metadata={}),
#             )
#         ],
#         include_source=False,
#     )


def create_hypothetical_requirements_nodes_for_privacy_policy_section(
    section: dict, hypothetical_requirements_vectors: list
):
    q = """
    MATCH (section:PrivacyPolicySection)
    WHERE section.id = $section_id
    UNWIND $hypothetical_requirements_vectors as h_req
    MERGE (h_req_node:PrivacyPolicyHypotheticalRequirement {text: h_req[0], embedding: h_req[1]})
    MERGE (section)-[:HAS_HYPOTHETICAL_REQUIREMENT]->(h_req_node)
    """
    graph.query(
        q,
        params={
            "section_id": section["node"]["id"],
            "hypothetical_requirements_vectors": hypothetical_requirements_vectors,
        },
    )


# TODO
def create_hypothetical_requirement_embedding_index(measure="cosine"):
    dimension = 1024
    embedding_name = "gte-large"
    q = """
    CALL db.index.vector.createNodeIndex($index_name, 'PrivacyPolicyHypotheticalRequirement', $embedding_name, $dimension, $measure)
    """
    graph.query(
        q,
        params={
            "embedding_name": embedding_name,
            "index_name": "hypothetical-requirement-embedding-index",
            "dimension": dimension,
            "measure": measure,
        },
    )


def hypothetical_requirement_similarity_search(
    vector: list,
    embedding_name="gte-large",
    limit=10,
):
    # q = f"""
    # CALL db.index.vector.queryNodes($index_name, $limit, $embedding)
    # YIELD node, score
    # WITH node, score
    # MATCH (section:PrivacyPolicySection)-[:HAS_HYPOTHETICAL_REQUIREMENT]->(node)
    # OPTIONAL MATCH (fulfilled_node:CodePiece)-[:IS_FULFILLED_BY]->(section)
    # OPTIONAL MATCH (unfulfilled_node:CodePiece)-[:IS_NOT_FULFILLED_BY]->(section)
    # RETURN node, score, section, fulfilled_node, unfulfilled_node
    # """
    q = f"""
    CALL db.index.vector.queryNodes($index_name, $limit, $embedding)
    YIELD node, score
    RETURN node, score
    """
    results = graph.query(
        q,
        params={
            "index_name": "hypothetical-requirement-embedding-index",
            "embedding": vector,
            "limit": limit,
        },
    )
    return list(sorted(results, key=lambda r: r["score"], reverse=True))


def create_privacy_policy_section_indexes():
    q = """
    CALL db.index.fulltext.createNodeIndex('PrivacyPolicySectionTextIndex', ['PrivacyPolicySection'], ['text'])
    """
