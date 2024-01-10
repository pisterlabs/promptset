from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain import OpenAI

class DocumentRetriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = OpenAI(temperature=0)
        print("DocumentRetriever Called...")

    def retrieve_documents(self, query, num_articles=None):
        metadata_field_info = [
            AttributeInfo(name="date_time", description="The date and time the news article was published", type="string"),
            AttributeInfo(name="headline", description="The headline of the news article", type="string"),
            AttributeInfo(name="link", description="The URL link to the original news article", type="string"),
        ]
        document_content_description = "Brief summary of a news article"
        
        retriever = SelfQueryRetriever.from_llm(
            self.llm,
            self.vectorstore,
            document_content_description,
            metadata_field_info,
            enable_limit=False,
            verbose=True,
            search_type="mmr",
            search_kwargs={'k': num_articles, 'lambda_mult': 0.25},
        )
        return retriever.get_relevant_documents(query)
