from langchain.schema import Document
from onepoint_document_chat.service.text_extraction import PAGE, FILE_NAME


def create_doc1():
    return Document(
        page_content="This is doc1", metadata={FILE_NAME: "file1", PAGE: [1]}
    )


def create_doc2():
    return Document(
        page_content="This is doc2", metadata={FILE_NAME: "file1", PAGE: [2]}
    )


def create_doc_long():
    return Document(
        page_content="""
GenAI and Explainable AI (xAI). Despite so many techniques and approaches "to guide and align the GenAI model outputs,” the fact is that GenAI models (naturally) still generate many unexpected, surprising, or totally wrong outputs. Increasingly, more than ever, I hear a lot of business people asking for “model explanations” and “interpretable models that can be explained.” Once the GenAI phase of hype & exploration fades away, we’ll see xAI becoming a priority. Here are a few notes on xAI:

Better tools for explainable AI. Dalex is an R & Python package that xrays any ML model and helps to explore and explain its behaviour, and helps to understand how complex models work. The philosophy behind DALEX explanations is described in this free e-book: Explanatory Model Analysis. Dalex incorporates the latest developments in Interpretable Machine Learning/ eXplainable AI. Checkout the repo, examples, and documentation here: DALEX- moDel Agnostic Language for Exploration and eXplanation
""",
        metadata={FILE_NAME: "file_long_1", PAGE: [1]},
    )


def create_list_simple1():
    return [create_doc1(), create_doc2()]


def create_single_long_doc_list():
    return [create_doc_long()]


def create_list_3():
    return create_list_simple1() + [create_doc_long()]
