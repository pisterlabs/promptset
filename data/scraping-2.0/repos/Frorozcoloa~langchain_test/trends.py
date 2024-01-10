from googlesearch import search
from typing import List, Dict
from newspaper import Article

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document


from config import codegpt_api_key, code_gpt_agent_id, codegpt_api_base
from utils import text2json, save_csv


def get_template() -> List[PromptTemplate]:
    """Returns a list of PromptTemplate objects with the following templates:"""

    def get_topic():
        """Returns a PromptTemplate object with the following template:"""
        template = """
        Given the following docs about a sports e-commerce, conduct an analysis of potential future trends.
        return a list of 10 topics.
        Output is a JSON list with the following format
        [
            {{"product_decription": "<product_decription>", "product_to_sell": "<product_to_sell1>"}},}}, 
            {{"product_decription": "<product_decription>", "product_to_sell": "<product_to_sell2>"}},}},
            ...
        ]
        {docs}
        """
        prompt = PromptTemplate(template=template, input_variables=["news"])
        return prompt

    def get_summary():
        """Returns a PromptTemplate object with the following template:"""
        template = """
        The following is a set of documents:

        {docs}

        Based on this list of docs, please identify the main themes 

        Helpful Answer:
        """
        prompt = PromptTemplate(template=template, input_variables=["news"])
        return prompt

    template_summary = get_summary()
    template_topic = get_topic()
    return [template_summary, template_topic]


def get_model(prompt_summary, prompt_topic) -> List[LLMChain]:
    """Returns a list of LLMChain objects"""
    llm = ChatOpenAI(
        openai_api_key=codegpt_api_key,
        openai_api_base=codegpt_api_base,
        model=code_gpt_agent_id,
    )

    def get_chain(llm: ChatOpenAI, template: PromptTemplate):
        """Returns a LLMChain object"""
        llm_chain = LLMChain(prompt=template, llm=llm)
        return llm_chain

    llm_summary = get_chain(llm, prompt_summary)
    llm_topic = get_chain(llm, prompt_topic)
    return [llm_summary, llm_topic]


def get_articles_trends(query: str = "Sports market trends", num_results: int = 50):
    """Found in google the articles related to the query and return a list of Document objects"""
    list_text = []
    for url in search(query, num_results=num_results):
        article = Article(url)
        article.download()
        article.parse()
        doc = Document(page_content=article.text, metadata={"source": url})
        list_text.append(doc)
    return list_text


def get_map_reduce(llm_summary: LLMChain):
    """Returns a summary of the list of documents"""
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_summary, document_variable_name="docs"
    )
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=llm_summary,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )
    return map_reduce_chain


def get_splitter():
    """Returns a CharacterTextSplitter object"""
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    return text_splitter


def get_summary_trends(llm_summary: LLMChain, list_docs: List[Document]) -> str:
    """Returns a summary of the list of documents"""
    map_reduce_chain = get_map_reduce(llm_summary)
    text_splitter = get_splitter()
    split_docs = text_splitter.split_documents(list_docs)
    text_summary = map_reduce_chain.run(split_docs)
    return text_summary


def get_topics(llm_topic: LLMChain, text_summary: str) -> str:
    """Returns a list of topics"""
    raw_topics = llm_topic.run(text_summary)
    topics = text2json(raw_topics)
    return topics


def get_analysis_trends(list_docs: list) -> List[Dict]:
    """Returns a list of topics, given a description of a product"""
    llm_summary, llm_topic = get_model(*get_template())
    text_summary = get_summary_trends(llm_summary, list_docs)
    topics = get_topics(llm_topic, text_summary)
    save_csv(topics, "trends")
    return topics


def example():
    """Example of use"""
    list_docs = get_articles_trends()
    topics = get_analysis_trends(list_docs)
    print(topics)


if __name__ == "__main__":
    example()
