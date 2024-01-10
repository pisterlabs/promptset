from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.document_loaders import PyPDFLoader


from app.model.chain_type import Chain_Type
from .open_ai import open_api_key, embedding

llm = OpenAI(openai_api_key=open_api_key, temperature=0.5)


def summery(pdf_location: str, chain_type: Chain_Type):
    loader = PyPDFLoader(pdf_location)
    pages = loader.load_and_split()
    summary_chain = load_summarize_chain(
        llm, chain_type=str(chain_type).split(".")[-1],
    )
    summarize_document_chain = AnalyzeDocumentChain(
        combine_docs_chain=summary_chain)
    return summarize_document_chain.run("".join(
        [t.page_content for t in pages]
    ))


def summery_from_text(text: str, chain_type: Chain_Type):
    summary_chain = load_summarize_chain(
        llm, chain_type=str(chain_type).split(".")[-1],
    )
    summarize_document_chain = AnalyzeDocumentChain(
        combine_docs_chain=summary_chain)
    return summarize_document_chain.run(text)
