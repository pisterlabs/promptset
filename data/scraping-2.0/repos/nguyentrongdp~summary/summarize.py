from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate, Prompt
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from llm.OpenRouterLLM import OpenRouterLLM
from langchain.chains.summarize import load_summarize_chain

load_dotenv()

# This controls how each document will be formatted. Specifically,
# it will be passed to `format_document` - see that function for more
# details.
document_prompt = PromptTemplate(
    input_variables=["page_content"],
    template="{page_content}"
)

# Define prompt
prompt_template = """
Summarize this document (only the summary of the response is included. Eliminate redundant sentences: Summary,...):
"{text}"
"""

prompt = PromptTemplate.from_template(prompt_template)

# Define LLM chain
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k-0613")
# llm = OpenRouterLLM(n=1, model='mistralai/mixtral-8x7b-instruct')
# llm = OpenRouterLLM(n=1, model='mistralai/mistral-7b-instruct')
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Define StuffDocumentsChain
stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name="text"
)


def summarize_doc(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    res = stuff_chain.run(docs)

    return res


if __name__ == '__main__':
    summarize_text = summarize_doc('./docs/15440478.2020.1818344.pdf')
    print(summarize_text)
