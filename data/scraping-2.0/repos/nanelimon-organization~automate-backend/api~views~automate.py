from openai import AzureOpenAI
from qdrant_client import QdrantClient
from decouple import Config, RepositoryEnv
from jinja2 import Environment, FileSystemLoader

class AutoHackPromptBuilder:
    """
    A class specifically designed for the Borusan AutoHack competition, focused on 
    generating system prompts using Retrieval-Augmented Generation (RAG).

    This class encapsulates the process of embedding queries, retrieving context, and 
    generating prompts for the specific needs of the AutoHack competition. It is tailored 
    for AI applications that require dynamic and context-sensitive prompt generation.

    Attributes
    ----------
    embedding_client : AzureOpenAI
        Client for embedding generation.
    search_client : QdrantClient
        Client for conducting search queries.
    template_env : Environment
        Jinja2 environment for loading templates.
    collection_name : str
        The name of the collection to be used in the search.
    """

    def __init__(self, collection_name, template_dir='api/templates/', template_file='prompt_template.j2'):
        self.config = Config(RepositoryEnv('.env'))
        self.embedding_client = AzureOpenAI(
            api_key=self.config('AZURE_OPENAI_EMBD_API_KEY'),
            api_version="2023-05-15",
            azure_endpoint=self.config('AZURE_OPENAI_EMBD_ENDPOINT')
        )
        self.search_client = QdrantClient(
            url=self.config('QDRANT_URL'),
            api_key=self.config('QDRANT_API_KEY')
        )
        self.collection_name = collection_name
        self.embedding_model_name = "text-embedding-ada-002"
        self.template_env = Environment(loader=FileSystemLoader(template_dir))
        self.template_file = template_file

    def retrieve_context(self, query, top_k=3):
        embedded_query = self.embedding_client.embeddings.create(
            input=query, model=self.embedding_model_name).data[0].embedding

        search_results = self.search_client.search(
            collection_name=self.collection_name,
            query_vector=embedded_query, 
            limit=top_k
        )

        context = ' '.join(
            [' '.join(point.payload['text'].split()) for point in search_results]
        )

        return context

    def generate_system_prompt(self, question):
        context = self.retrieve_context(question, top_k=3)
        template = self.template_env.get_template(self.template_file)
        return template.render(documentation_context=context)
