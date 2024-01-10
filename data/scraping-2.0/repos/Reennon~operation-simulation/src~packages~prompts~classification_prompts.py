from langchain import PromptTemplate


class ClassificationPromptWithoutMemory:
    template = """Given a ("QUERY") and ("API_DOCS") decide whether the ("QUERY") is a conversation query or
api-request query and output "1" or "0".
The conversation query implies clarifications about ("API_DOCS") or about Military Operation Simulations.
The api documentation ("API_DOCS") clarifies how Military Operations Simulator works.
The api-request query ("QUERY") implies straightforward request to API.
If you believe that the ("QUERY") is conversation query type "1".
If you believe that the ("QUERY") is api-request query type "0".
If you don't know whether the query ("QUERY") is conversational or api-request, type "1".
If query ("QUERY") is not related to the ("API_DOCS") or api-request, type "1".
ALWAYS return true or false, true corresponding to conversation query ("QUERY"), and false corresponding to api-request 
query ("QUERY").
RETURN only "1" or "0" with no additional characters.

EXAMPLES
--------
QUERY: What is a Military Operation Simulation?
API_DOCS: {api_docs}
FINAL ANSWER: true

QUERY: Simulate attack on group alpha on right flang being safe
API_DOCS: {api_docs}
FINAL ANSWER: false
---------------
END OF EXAMPLES

QUERY: {query}
API_DOCS: {api_docs}
FINAL ANSWER:"""

    QUERY_CLASSIFICATION_PROMPT = PromptTemplate(
        template=template,
        input_variables=['query', 'api_docs']
    )


class ClassificationPromptWithMemory:
    template = """Given a ("QUERY"), ("CHAT_MEMORY") and ("API_DOCS") decide whether the ("QUERY") is a 
conversation query or api-request query, being aware of the context of the conversation using history ("CHAT_MEMORY").
The conversation query implies clarifications about ("API_DOCS") or about Military Operation Simulations. 
The api-request query ("QUERY") implies straightforward request to API.
The api documentation ("API_DOCS") clarifies how Military Operations Simulator works.
If you don't know whether the query ("QUERY") is conversational or api-request, type true.
If query ("QUERY") is not related to the ("API_DOCS") or api-request, type true.
ALWAYS return true or false, true corresponding to conversation query ("QUERY"), and false corresponding to api-request 
query ("QUERY").

EXAMPLES
--------
QUERY: What is a Military Operation Simulation?
API_DOCS: {api_docs}
FINAL ANSWER: true

QUERY: Simulate attack on group alpha on right flang being safe
API_DOCS: {api_docs}
FINAL ANSWER: false
---------------
END OF EXAMPLES

QUERY: {query}
CHAT_MEMORY: {chat_memory}
API_DOCS: {api_docs}
FINAL ANSWER:"""

    QUERY_CLASSIFICATION_PROMPT = PromptTemplate(
        template=template,
        input_variables=['query', 'api_docs', 'chat_memory']
    )

