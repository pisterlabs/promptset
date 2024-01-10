from knowledge_graph_clients.knowledge_graph_client import KnowledgeGraphClient, Node
from pymilvus import Collection
from agentware.agent_logger import Logger
import unittest
import openai

logger = Logger()
CFG = None


def get_embedding(text: str):
    text = text.replace("\n", " ")
    logger.debug(f"Getting embedding of: {text}")
    vector = openai.Embedding.create(input=[text], model="text-embedding-ada-002")[
        'data'][0]['embedding']
    return vector


class TestEmbeddedKnowledgeGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # This method will be called once before running all tests
        cls.graph = KnowledgeGraphClient(CFG)
        cls.default_collection_name = "test_kg_node_index_1"

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        # Delete collection
        c = Collection(self.default_collection_name)
        c.drop()
        # Clean up knowledge graph
        with self.graph.neo4j_driver.session() as session:
            query = "MATCH (n {collection_name: $collection_name}) DETACH DELETE n"
            session.run(query, collection_name=self.default_collection_name)
        return super().tearDown()

    def test_insert_relation(self):
        node1 = Node("panda")
        node2 = Node("animal")
        triplet = (node1, "is a kind of", node2)
        self.graph.insert_relation(
            triplet, self.default_collection_name, get_embedding)
        # Here you can assert conditions, for example checking that the triplet exists in the graph
        node1_id_in_vector_store = self.graph.vector_db_client.get_node_id(
            node1, self.default_collection_name)
        logger.debug(
            f"node 1 {node1} id in vector db is {node1_id_in_vector_store}")
        with self.graph.neo4j_driver.session() as session:
            node1_id_in_kg = session.run("""
                    MATCH (n:Node {name: $node, collection_name: $collection_name})
                    RETURN n.vector_store_id AS vector_store_id
                    """, {"node": node1.name, "collection_name": self.default_collection_name}).single()['vector_store_id']
            logger.debug(f"node 1 id in kg is {node1_id_in_kg}")
        assert node1_id_in_kg == node1_id_in_vector_store

    def test_keyword_search(self):
        self.graph.insert_relation(
            (Node("panda"), "is a kind of", Node("animal")), self.default_collection_name, get_embedding)
        self.graph.insert_relation(
            (Node("mark zuckerberg"), "is the founder of",
             Node("meta")), self.default_collection_name, get_embedding)
        paths1 = self.graph.keyword_search(
            get_embedding("bear"), self.default_collection_name)
        assert paths1[0] == 'panda is a kind of animal'

        paths2 = self.graph.keyword_search(
            get_embedding("silicon valley"), self.default_collection_name)
        logger.debug(f"paths2 is {paths2}")
        assert paths2[0] == 'mark zuckerberg is the founder of meta'
        # Here you can assert conditions, for example checking that the paths are what you expect

    # def test_3_delete_relation(self):
    #     triplet = ("panda", "is protected by", "human")
    #     self.graph.delete_relation(triplet, self.collection)
    #     # Here you can assert conditions, for example checking that the triplet no longer exists in the graph

    # def test_4_keyword_search_after_deletion(self):
    #     keyword = "animal china"
    #     paths = self.graph.keyword_search(keyword, self.collection)
    #     # Here you can assert conditions, for example checking that the paths are what you expect after deletion


if __name__ == "__main__":
    unittest.main()
