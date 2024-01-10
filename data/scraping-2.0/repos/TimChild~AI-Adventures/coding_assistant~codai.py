from dataclasses import dataclass, field
import os

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import ReadTheDocsLoader
import langchain.agents


with open("../API_KEY", "r") as f:
    os.environ["OPENAI_API_KEY"] = f.read()


llm = ChatOpenAI(temperature=0.0)

DECIDER_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "text_prompt_input",
        "existing_code_input",
        "error_messages_input",
        "project_context_input",
    ],
    template="""You are an AI with a single responsibility: to decide which type of response is best based on the following inputs.

The existing code is:
-----
{existing_code_input}
-----

The error messages are:
-----
{error_messages_input}
-----

The current project context is:
-----
{project_context_input}
-----

The user input is:
-----
{text_prompt_input}
-----

You should decide which type of response is best based on the following rules:
- Decide between these possible response types: 
    - "new code" -- use this when the user is asking for mostly new code
    - "modify code" -- use this when the user is asking for a modification to existing code or to solve an error message
    - "get more info"-- use this when there is not enough information to give a good response to the user's request (i.e. if the user input is too vague)
    - "general" -- use this when the user input is asking for explanation rather than code, or if the other response types are not appropriate
- Give your answer in the following format (a JSON object):
    -----
    {{response_type: <response_type>, reason: <reason>}}
    -----
- The reason should be a short string that explains why you chose the response type
- The response type should be one of response types (e.g. "new code", "modify code", "get more info", or "general")
""",
)

PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "text_prompt_input",
        "existing_code_input",
        "error_messages_input",
        "project_context_input",
    ],
    template="""You are a python coding assistant and will be using the following information to help a user with their code related question.
                             
The existing code is:
-----
{existing_code_input}
-----

The error messages are:
-----
{error_messages_input}
-----

The current project context is:
-----
{project_context_input}
-----

The user input is:
-----
{text_prompt_input}
-----
                                 
In your answer you should always follow these rules:
- Give your answer with Markdown formatting (especially for code blocks)
- Use the latest python programming techniques and best practices
- Use the latest python libraries when appropriate
- Always include a google style docstring for functions
- Address any error messages that are present
- Use the existing code as a starting point if present
- Include type hints where appropriate
- If the answer is a simple piece of code, only include the code and not the explanation

Based on the information above, give a response that will satisfy the user's request.
""",
)


@dataclass
class SingleResponse:
    code: str
    summaries: str
    thought_process: str


@dataclass
class SingleRequest:
    text_input: str
    existing_code_input: str
    project_context_input: str
    error_messages_input: str


def generate_response(
    text_input, existing_code_input, project_context_input, error_messages_input
) -> SingleResponse:
    cody = CodAI()
    return cody._generate_general_response(
        SingleRequest(text_input, existing_code_input, project_context_input, error_messages_input)
    )


class CodAI:
    def __init__(self):
        self.request_history = []
        self.response_history = []

    def get_response(self, single_request: SingleRequest) -> SingleResponse:
        """Get a response to a single request."""
        response_type = self._decide_response_type(single_request)
        if response_type == "new code":
            return self._generate_new_code_response(single_request)
        elif response_type == "modify code":
            return self._generate_modify_code_response(single_request)
        elif response_type == "get more info":
            return self._generate_require_more_info_response(single_request)
        elif response_type == "general":
            return self._generate_general_response(single_request)
        else:
            raise ValueError(f"Invalid response type: {response_type}")

    def _decide_response_type(self, single_request: SingleRequest) -> str:
        chain = LLMChain(llm=llm, prompt=DECIDER_PROMPT_TEMPLATE)
        response = chain.run(
            {
                "text_prompt_input": single_request.text_input,
                "existing_code_input": single_request.existing_code_input,
                "project_context_input": single_request.project_context_input,
                "error_messages_input": single_request.error_messages_input,
            }
        )

        return response

    def _generate_general_response(
        self, single_request: SingleRequest
    ) -> SingleResponse:
        chain = LLMChain(llm=llm, prompt=PROMPT_TEMPLATE)
        response = chain.run(
            {
                "text_prompt_input": single_request.text_input,
                "existing_code_input": single_request.existing_code_input,
                "project_context_input": single_request.project_context_input,
                "error_messages_input": single_request.error_messages_input,
            }
        )
        # TESTING
        response = self._decide_response_type(single_request) + "\n\n" + response
        return SingleResponse(code=response, summaries="", thought_process="")

    def _generate_new_code_response(
        self, single_request: SingleRequest
    ) -> SingleResponse:
        # Implement logic to generate a new code response based on the input
        # Return a SingleResponse object
        pass

    def _generate_modify_code_response(
        self, single_request: SingleRequest
    ) -> SingleResponse:
        # Implement logic to generate a modify code response based on the input
        # Return a SingleResponse object
        pass

    def _generate_require_more_info_response(
        self, single_request: SingleRequest
    ) -> SingleResponse:
        # Implement logic to generate a require more info response based on the input
        # Return a SingleResponse object
        pass
