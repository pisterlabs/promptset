from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from .prompt import QUESTION_VARIANT_PROMPT

class RagFusion:
    """Class for generating multiple questions from an example question, querying a vectorstore,
    and ranking the results."""

    def __init__(self, openai_api_key, vectordb):
        """
        Initialize the RagFusion class.

        Args:
            openai_api_key (str): The API key for OpenAI.
            vectordb (object): The vector database object.
        """
        self.OPENAI_API_KEY = openai_api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)
        self.db = vectordb

    def generate_question_variants(self, question):
        """
        Generate multiple variants of a given question.

        Args:
            question (str): The original question.

        Returns:
            list: A list of question variants.
        """
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.6, openai_api_key=self.OPENAI_API_KEY, streaming=True)
        formatted_template = QUESTION_VARIANT_PROMPT.format_messages(text=question)
        results = llm(formatted_template)
        return results.content.split('\n')

    def db_lookup(self, question, search_kwargs=""):
        """
        Perform a direct lookup against the vectorstore.

        Args:
            question (str): The question to search for.
            search_kwargs (str, optional): Additional search parameters. Defaults to "".

        Returns:
            list: A list of search results with relevance scores.
        """
        if not question:
            return
        emb = self.embeddings.embed_query(question)
        # Note that this method is a LangChain-specific method for Chromadb. Will need to change if I start supporting other vectorstores
        return self.db.similarity_search_by_vector_with_relevance_scores(emb, k=4, search_kwargs=search_kwargs)
        
    def reciprocal_rank_fusion(self, search_results_dict, k=60):
        """
        Perform a Reciprocal Rank Fusion (RRF) on the search results. RRF  
        combines the ranks of the results from multiple search queries to produce a single 
        ranking list. 

        The choice of 60 as a constant in the Reciprocal Rank Fusion (RRF) formula is somewhat 
        arbitrary and is often used as a default value in Information Retrieval tasks. 
        It's a balance between giving enough weight to high-ranked items and not overly penalizing 
        lower-ranked ones.

        Args:
            search_results_dict (dict): The search results dictionary.
            k (int, optional): The rank parameter. Defaults to 60.

        Returns:
            dict: A dictionary of fused scores.
        """
        fused_scores = {}
        doc_ranks = {}
        for query, search_results_list in search_results_dict.items():
            fused_scores[query] = {}
            # Filter out any empty elements that have crept in
            if search_results_list is None:
                continue
            for doc, score in sorted(search_results_list, key=lambda x: x[1], reverse=True):
                doc_name = doc.metadata['filename']
                doc_ranks[doc_name] = doc_ranks.get(doc_name, 0) + 1
                rank = doc_ranks[doc_name]
                fused_scores[query][doc_name] = 1 / (rank + k)

            fused_scores[query] = {doc: score for doc, score in sorted(fused_scores[query].items(), key=lambda x: x[1], reverse=True)}

        return fused_scores