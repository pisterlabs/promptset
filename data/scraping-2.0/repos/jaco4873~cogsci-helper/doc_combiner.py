'''
Doc combiner function to combine retrieved documents
'''

from langchain.prompts import PromptTemplate
from langchain.schema import format_document


default_doc_prompt = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=default_doc_prompt, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)