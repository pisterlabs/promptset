from langchain.prompts import PromptTemplate

from src.packages.prompts.api_url_prompts import APIURLPromptWithHistory, APIURLPromptWithoutHistory


class APIResponsePromptWithHistory:
    template = APIURLPromptWithHistory.template + """ The endpoint ("API_URL") was used
    
Given ("API_DOCS"), ("API_URL"), ("API_RESPONSE"), ("QUERY") and ("CHAT_HISTORY"), generate summarization 
("SUMMARIZATION") of this response, while considering context of conversation ("CHAT_HISTORY") and original user 
question ("QUERY")

QUERY: {query}
CHAT_HISTORY: {chat_history}
API_URL: {api_url}
API_DOCS: {api_docs}
SUMMARIZATION:"""

    API_URL_PROMPT = PromptTemplate(
        input_variables=[
            "api_docs",
            'api_url',
            'api_response'
            'query',
            'chat_history',
        ],
        template=template,
    )


class APIResponsePromptWithoutHistory:
    template = APIURLPromptWithoutHistory.template + """ The endpoint ("API_URL") was used
    
Given ("API_DOCS"), ("API_URL"), ("API_RESPONSE") and ("QUERY"), generate summarization ("SUMMARIZATION") of this 
response, while considering context of original user question ("QUERY")

QUERY: {query}
API_URL: {api_url}
API_DOCS: {api_docs}
SUMMARIZATION:"""

    API_URL_PROMPT = PromptTemplate(
        input_variables=[
            "api_docs",
            'api_url',
            'api_response'
            'query',
        ],
        template=template,
    )
