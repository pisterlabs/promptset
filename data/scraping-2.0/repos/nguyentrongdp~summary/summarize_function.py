from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from tools.summary import summary_tool
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage


import langchain

langchain.debug = True

load_dotenv()

# This controls how each document will be formatted. Specifically,
# it will be passed to `format_document` - see that function for more
# details.
# document_prompt = PromptTemplate(
#     input_variables=["page_content"],
#     template="{page_content}"
# )

# Define prompt
prompt_template = """
Summarize this document (only the summary of the response is included. Eliminate redundant sentences: Summary,...):
"{input}"
"""

# prompt = PromptTemplate.from_template(
#     prompt_template, input_variables=["text"])

prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(
            content=(
                "You are a wise AI that can summarize all scientific articles in a simple way. Please summarize and provide 50 main keywords and category of the article."
            )
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template(prompt_template),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
# Define LLM chain
llm = ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview")
# llm = OpenRouterLLM(n=1, model='mistralai/mixtral-8x7b-instruct')
# llm = OpenRouterLLM(n=1, model='mistralai/mistral-7b-instruct')
# llm_chain = LLMChain(llm=llm, prompt=prompt)

# Define StuffDocumentsChain
# stuff_chain = StuffDocumentsChain(
#     llm_chain=llm_chain,
#     document_prompt=document_prompt,
#     document_variable_name="text"
# )

tools = [
    summary_tool
]

memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True
)

agent = OpenAIFunctionsAgent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(
    agent=agent,
    # verbose=True,
    tools=tools,
    memory=memory,
)


def summarize_doc(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    docs = ' '.join(list(map(lambda doc: doc.page_content, docs)))

    return agent_executor.run(docs)


if __name__ == '__main__':
    summarize_text = summarize_doc('./docs/15440478.2020.1818344.pdf')
    print(summarize_text)
