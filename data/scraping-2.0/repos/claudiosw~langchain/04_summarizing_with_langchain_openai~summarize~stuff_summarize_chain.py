from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_base import llm


def get_stuff_summarize_chain(text):
    docs = [Document(page_content=text)]


    template = '''Write a concise and short summary of the following text.
    TEXT: `{text}`
    '''
    prompt = PromptTemplate(
        input_variables=['text'],
        template=template
    )

    chain = load_summarize_chain(
        llm,
        chain_type='stuff',
        prompt=prompt,
        verbose=False
    )
    output_summary = chain.run(docs)
    return output_summary
