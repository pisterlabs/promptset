from langchain.llms import OpenAI
from indexer import product_descriptions

llm = OpenAI()


class Agent:
    """Agent that performs an atomic action
    Inputs:
    - purpose: str      # eg: Given the text, classify into one of following three categories.
    - evidence_to_aid_purpose: str     # key value pairs or equivalent of evidence to act on
    """

    def __init__(self, purpose: str, evidence_to_aid_purpose: str) -> None:
        self.purpose = purpose
        self.evidence_to_aid_purpose = evidence_to_aid_purpose
        pass

    def __call__(self, query: str) -> str:
        constructed_query = self.construct_query(query=query)
        return llm(constructed_query)

    def construct_query(self, query: str) -> str:
        separator = "\n\n"
        constructed_query = (
            self.purpose
            + separator
            + self.evidence_to_aid_purpose.format(
                product_descriptions=product_descriptions, query=query
            )
        )
        return constructed_query


llm = OpenAI()

classification_agent = Agent(
    purpose="Classify the user query into a product",
    evidence_to_aid_purpose="""
    Restrict your classifications to one of the following products only: "Primescan Connect", "CEREC Primemill", "CEREC SW 5", "IFU Primescan Connect DE", "None"

    Here are descriptions of each product to help with your classification:
    {product_descriptions}

    Use below evidence to classify.

    Query 1: How do I recalibrate my scanner?
    Product 1: Primescan Connect
    ###
    Query 2: My camera is not working, what should I do?
    Product 2: Primescan Connect
    ###
    Query 3: How do I change the filter bag and HEPA filter on my device?
    Product 3: CEREC Primemill
    ###
    Query 4: How do I add new devices to my software?
    Product 4: CEREC SW 5
    ###
    Query 5: How do I change the bedding on my mattress?
    Product 5: None
    ###
    Query 6: So führen Sie einen okklusalen Scan durch?
    Product 6: IFU Primescan Connect DE
    ###
    Query 7: Quadranten- und Vollkiefer-Scan?
    Product 7: IFU Primescan Connect DE
    ###
    Query 8: Dutch german
    Product 8: IFU Primescan Connect DE
    ###
    Query 9: How do I do something in english?
    Product 9: Primescan Connect
    ###
    Query 10: {query}
    Product 10:

    """,
)
