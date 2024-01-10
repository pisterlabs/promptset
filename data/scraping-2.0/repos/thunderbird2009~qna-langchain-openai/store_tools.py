from langchain.tools.base import BaseTool
from typing import Optional, Type, Any
import json
import sys
from langchain.vectorstores import FAISS
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.embeddings.openai import OpenAIEmbeddings

class ProdSearchTool(BaseTool):
    name = "prod_search"
    description = """Search product catalog. 
    Input: product name, category and description, etc. 
    Output: a list of products in JSon. Each product has the following fields: 
        category, subcategory, subcategory-link, product-link, name, price, description.
        Compose a snippet with product-link as a href for each product to show to customer.
    """
    docsearch: Optional[FAISS] = None

    def __init__(self, prod_embedding_store, embeddings, **data: Any) -> None:
        super().__init__(**data)
        self.docsearch = FAISS.load_local(
            folder_path=prod_embedding_store, embeddings=embeddings)

    def findProds(self, query) -> str:
        data_list = self.docsearch.similarity_search(query)
        # Define a mapping of old keys to new keys (including the NULL mappings)
        key_mapping = {
            'category-links': 'category',
            'subcategory-links': 'subcategory',
            'subcategory-links-href': 'subcategory-link',
            'product-links-href': 'product-link',
            'name': 'name',
            'price': 'price',
            'description': 'description'
        }
        updated_items = []
        for item in data_list:
            updated_item = {key_mapping.get(
                key, key): value for key, value in item.metadata.items() if key in key_mapping}
            updated_items.append(updated_item)

        json_data = {'type': 'prod_list', 'products': updated_items}
        return json.dumps(json_data)

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.findProds(query)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("ProdSearchTool does not support async")


class CustServiceTool(BaseTool):
    name = "customer_service"
    description = """General customer service that handles questions about buyer\'s store experience,
    such as account, user profile, order, payment, shipment, return, shopping cart, etc.
    Input: customer request.
    Output: Text relevant to the question."""
    faqsearch: Optional[FAISS] = None

    def __init__(self, faq_embedding_store, embeddings) -> None:
        super().__init__()
        self.faqsearch = FAISS.load_local(
            folder_path=faq_embedding_store, embeddings=embeddings)

    def findFAQs(self, query) -> str:
        data_list = self.faqsearch.similarity_search_with_relevance_scores(
            query, k=1)
        if len(data_list) == 0 or data_list[0][1] < 0.5:
            json_data = {'type': 'final_msg',
                         'msg': 'Have not found an answer from our knowledge base. Will redirect you to an agent.'}
            return json.dumps(json_data)
        else:
            doc = data_list[0][0]
            json_data = {
                'type': 'kb_src', 'src': doc.metadata["source"], 'context': doc.page_content}
            return json.dumps(json_data)

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.findFAQs(query)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("CustServiceTool does not support async")


class DefaultTool(BaseTool):
    name = "default_tool"
    description = """Default tool to handle all questions or requests that can not be handled by
    other tools.
    Input: customer request.
    Output: final answer."""
    faqsearch: Optional[FAISS] = None

    DEFAULT_TOOL_MSG = """I am a chatbot that can handle product and store customer service questions. 
    Your question seems to be outside my scope. Could you rephrase it for me to understand better, 
    or ask a different question? Thx!
    """
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        json_data = {'type': 'final_msg', 'msg': self.DEFAULT_TOOL_MSG}
        return json.dumps(json_data)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("DefaultTool does not support async")
    

from abc import ABC, abstractmethod

class ChatbotBase(ABC):
    @abstractmethod
    def answer(self, user_msg) -> str:
        pass
