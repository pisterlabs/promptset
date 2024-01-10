# This plugin uses Azure Cognitive Search to search a knowledge base for a query.

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from dotenv import dotenv_values
from semantic_kernel.orchestration.sk_context import SKContext
from semantic_kernel.skill_definition import sk_function, sk_function_context_parameter
import openai


class AzureCognitiveSearch:
    def __init__(
        self,
        endpoint=None,
        key=None,
        index_name=None,
        content_field=None,
        reference_field=None,
        top=5,
    ):
        # load config
        config = dotenv_values("../.env")
        self.endpoint = (
            endpoint or f"https://{config['AZURE_SEARCH_SERVICE']}.search.windows.net/"
        )
        self.key = key or config["AZURE_SEARCH_KEY"]
        self.index_name = index_name or config["AZURE_SEARCH_INDEX"]
        self.content_field = content_field or config["AZURE_SEARCH_CONTENT_FIELD"]
        self.reference_field = reference_field or config["AZURE_SEARCH_REFERENCE_FIELD"]
        self.top = top
        # OpenAI for vector search
        openai.api_base = config["AZURE_OPENAI_ENDPOINT"]
        openai.api_version = "2022-12-01"
        openai.api_type = "azure"
        openai.api_key = config["AZURE_OPENAI_API_KEY"]
        self._openai_embedding_model = config["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

        print("Loaded Azure Cognitive Search Plugin")

    def get_search_client(self):
        """Create a SearchClient to query the index."""
        return SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(self.key),
        )

    async def create_embedding(self, text, openai_embedding_model):
        """Create an embedding for a given text using OpenAI embedding model."""
        try:
            embedded_text = (
                await openai.Embedding.acreate(
                    input=text, deployment_id=openai_embedding_model
                )
            )["data"][0]["embedding"]
        except Exception as e:
            print(f"Error creating embedding for text: {text} with error: {e}")
            embedded_text = None
        return embedded_text

    async def remove_newlines(self, s):
        s = s.replace("\n", " ").replace("\r", " ")
        return s

    @sk_function(
        description="Given a query, search an index and return the results.",
        name="search",
    )
    @sk_function_context_parameter(
        name="input",
        description="The query to search for in the knowledge base",
    )
    async def search(self, context: SKContext) -> str:
        query = context["input"]
        embedded_query = await self.create_embedding(
            query, self._openai_embedding_model
        )
        search_client = self.get_search_client()
        async with search_client:
            r = await search_client.search(
                search_text=query,
                top=self.top,
                vector_fields="content_vector",
                vector=embedded_query,
                top_k=10,
            )
            results = [
                doc[self.reference_field]
                + ": "
                + await self.remove_newlines(doc[self.content_field])
                async for doc in r
            ]
            content = "\n".join(results)

        context["search_result"] = "\nSOURCES:\n" + content

        return content
