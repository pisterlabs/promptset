from langchain.graphs import Neo4jGraph


def connect_neo_graph(url: str, username: str, password: str) -> Neo4jGraph:
    return Neo4jGraph(url=url, username=username, password=password)


def create_vector_index(driver: Neo4jGraph, dimension: int) -> None:
    index_query = "CALL db.index.vector.createNodeIndex('top_questions', 'Question', 'embedding', $dimension, 'cosine')"
    try:
        driver.query(index_query, {"dimension": dimension})
    except:  # Already exists
        pass
    index_query = "CALL db.index.vector.createNodeIndex('top_answers', 'Answer', 'embedding', $dimension, 'cosine')"
    try:
        driver.query(index_query, {"dimension": dimension})
    except:  # Already exists
        pass

    