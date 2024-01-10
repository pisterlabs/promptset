from langchain.prompts import PromptTemplate


class APIURLPromptWithHistory:
    template = """Given ("API_DOCS"), ("QUERY") and ("CHAT_HISTORY"), generate the full API url ("API_URL") to call for 
answering the user question ("QUERY"), while considering context of conversation ("CHAT_HISTORY").
Build the API url in order to get a response that is as short as possible, while still getting the necessary information 
to answer the question. Pay attention to deliberately exclude any unnecessary pieces of data in the API call.

QUERY: {query}
CHAT_HISTORY: {chat_history}
API_DOCS: {api_docs}
API_URL:"""

    API_URL_PROMPT = PromptTemplate(
        input_variables=[
            'query',
            'chat_history',
            "api_docs",
        ],
        template=template,
    )


class APIURLPromptWithoutHistory:
    template = """Given ("API_DOCS") and ("QUERY"), generate the full API url ("API_URL") to call for 
answering the user question ("QUERY").
Build the API url in order to get a response that is as short as possible, while still getting the necessary information 
to answer the question. Pay attention to deliberately exclude any unnecessary pieces of data in the API call.

QUERY: {query}
API_DOCS: {api_docs}
API_URL:"""

    API_URL_PROMPT = PromptTemplate(
        input_variables=[
            'query',
            "api_docs",
        ],
        template=template,
    )
