import pandas as pd
import os
import demjson3

from langchain.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.llms.together import Together
from utils.custom_langchain import NewCohereEmbeddings, NewPGVector
from langchain.chains.openai_functions import (
    create_structured_output_chain,
)
import tiktoken

import utils.custom_langchain as clc
import utils.db as db
import utils.prompts as ps
import utils.app_utils as au

CONNECTION_STRING = (
    f"postgresql+psycopg2://{db.db_params['user']}:{db.db_params['password']}"
    f"@{db.db_params['host']}:{db.db_params['port']}/{db.db_params['dbname']}"
)

token_encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

llm_map = {
    "GPT-3.5-Turbo-JSON": ChatOpenAI(
        model_name="gpt-3.5-turbo-1106", temperature=0.1
    ).bind(response_format={"type": "json_object"}),
    "GPT-3.5-Turbo": ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0.1),
    "GPT-3.5-Turbo-HT": ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0.8),
    "GPT-4": ChatOpenAI(model_name="gpt-4", temperature=0.1),
    "GPT-4-Turbo": ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.1),
    "GPT-4-Turbo-JSON": ChatOpenAI(
        model_name="gpt-4-1106-preview", temperature=0.1
    ).bind(response_format={"type": "json_object"}),
    "mixtral": Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.1,
        max_tokens=4000,
        together_api_key=os.getenv("TOGETHER_API_KEY"),
    ),
    "local": ChatOpenAI(model_name="local", temperature=0.1),
}


def validate_openai_env():
    """ Validate that the API base is not set to local."""
    api_base = os.environ.get("OPENAI_API_BASE", "")
    false_base = "http://localhost:1234/v1"
    assert api_base != false_base, "API base is not set to local."

def initialize_retriever(collection_name):
    """Initialize retriever for GPT maestro."""
    if collection_name == "arxiv_vectors_cv3":
        embeddings = NewCohereEmbeddings(
            cohere_api_key=os.getenv("COHERE_API_KEY"), model="embed-english-v3.0"
        )
    elif collection_name == "arxiv_vectors":
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=os.getenv("HUGGINGFACE_API_KEY"), model_name="thenlper/gte-large"
        )
    else:
        raise ValueError(f"Unknown collection name: {collection_name}")

    store = NewPGVector(
        collection_name=collection_name,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 20})

    compressor = CohereRerank(
        top_n=10, cohere_api_key=os.getenv("COHERE_API_KEY"), user_agent="llmpedia"
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever


def create_rag_context(parent_docs):
    """Create RAG context for LLM, including text excerpts, arxiv_codes,
    year of publication and citation counts."""
    rag_context = ""
    for idx, doc in parent_docs.iterrows():
        arxiv_code = doc["arxiv_code"]
        year = doc["published"]
        citation_count = doc["citation_count"]
        text = "..." + doc["text"] + "..."
        rag_context += (
            f"arxiv:{arxiv_code} ({year}, {citation_count} citations)\n\n{text}\n\n"
        )

    return rag_context


def query_llmpedia(question: str, collection_name, model="GPT-3.5-Turbo"):
    """Query LLMpedia via LLMChain."""
    rag_prompt_custom = ChatPromptTemplate.from_messages(
        [
            ("system", ps.VS_SYSYEM_TEMPLATE),
            ("human", "{question}"),
        ]
    )

    rag_llm_chain = LLMChain(
        llm=llm_map[model], prompt=rag_prompt_custom, verbose=False
    )
    compression_retriever = initialize_retriever(collection_name)
    child_docs = compression_retriever.invoke(question)

    ## Map to parent chunk (for longer context).
    child_docs = [doc.metadata for doc in child_docs]
    child_ids = [(doc["arxiv_code"], doc["chunk_id"]) for doc in child_docs]
    parent_ids = db.get_arxiv_parent_chunk_ids(child_ids)
    parent_docs = db.get_arxiv_chunks(parent_ids, source="parent")
    parent_docs["published"] = pd.to_datetime(parent_docs["published"]).dt.year
    parent_docs.sort_values(
        by=["published", "citation_count"], ascending=False, inplace=True
    )
    parent_docs.reset_index(drop=True, inplace=True)
    parent_docs = parent_docs.head(5)

    ## Create custom prompt.
    rag_context = create_rag_context(parent_docs)
    res = rag_llm_chain.run(context=rag_context, question=question)
    res_response = res.split("Response\n")[1].split("###")[0].strip()
    content = au.add_links_to_text_blob(res_response)

    return content


def summarize_doc_chunk(paper_title: str, document: str, model="local"):
    """Summarize a paper by segments."""
    summarizer_prompt = ChatPromptTemplate.from_messages(
        [("system", ps.SUMMARIZE_BY_PARTS_TEMPLATE)]
    )
    chain = LLMChain(llm=llm_map[model], prompt=summarizer_prompt, verbose=False)
    summary = chain.run({"paper_title": paper_title, "content": document})
    return summary


def verify_llm_paper(paper_content: str, model="GPT-3.5-Turbo-JSON"):
    """Verify if a paper is about LLMs via LLMChain."""
    llm_paper_check_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ps.LLM_PAPER_CHECK_TEMPLATE),
            ("human", "{paper_content}"),
            ("human", ps.LLM_PAPER_CHECK_FMT_TEMPLATE),
        ]
    )
    llm_chain = LLMChain(
        llm=llm_map[model], prompt=llm_paper_check_prompt, verbose=False
    )
    is_llm_paper = llm_chain.run(paper_content=paper_content)
    is_llm_paper = is_llm_paper.replace("\n", "")
    is_llm_paper = demjson3.decode(is_llm_paper)
    return is_llm_paper


def review_llm_paper(paper_content: str, model="GPT-3.5-Turbo"):
    """Review a paper via LLMChain."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ps.SUMMARIZER_SYSTEM_PROMPT),
            ("human", ps.SUMMARIZER_HUMAN_REMINDER),
        ]
    )
    parser = clc.CustomFixParser(pydantic_schema=ps.PaperReview)
    chain = create_structured_output_chain(
        ps.PaperReview, llm_map[model], prompt, output_parser=parser, verbose=False
    )
    review = chain.run(paper_content=paper_content)
    return review


def convert_notes_to_narrative(paper_title, notes, model="GPT-3.5-Turbo"):
    """Convert notes to narrative via LLMChain."""
    narrative_prompt = ChatPromptTemplate.from_messages(
        [("system", ps.NARRATIVE_SUMMARY_PROMPT)]
    )
    narrative_chain = LLMChain(llm=llm_map[model], prompt=narrative_prompt)
    narrative = narrative_chain.run(
        {"paper_title": paper_title, "previous_notes": notes}
    )
    return narrative


def copywrite_summary(paper_title, narrative, model="GPT-3.5-Turbo"):
    """Copywrite a summary via LLMChain."""
    copywriting_prompt = ChatPromptTemplate.from_messages(
        [("system", ps.COPYWRITER_PROMPT)]
    )
    copywriting_chain = LLMChain(llm=llm_map[model], prompt=copywriting_prompt)
    copywritten = copywriting_chain.run(
        {"paper_title": paper_title, "previous_summary": narrative}
    )
    return copywritten


def convert_notes_to_markdown(paper_title, notes, model="GPT-3.5-Turbo"):
    """Convert notes to markdown via LLMChain."""
    markdown_prompt = ChatPromptTemplate.from_messages(
        [("system", ps.MARKDOWN_PROMPT)]
    )
    markdown_chain = LLMChain(llm=llm_map[model], prompt=markdown_prompt)
    markdown = markdown_chain.run(
        {"paper_title": paper_title, "previous_notes": notes}
    )
    return markdown


def summarize_title_in_word(title, model="GPT-3.5-Turbo-HT"):
    """Summarize a title in a few words via LLMChain."""
    title_summarizer_prompt = ChatPromptTemplate.from_messages(
        [("system", ps.TITLE_SUMMARIZER_PROMPT)]
    )
    title_summarizer_chain = LLMChain(
        llm=llm_map[model], prompt=title_summarizer_prompt
    )
    keyword = title_summarizer_chain.run({"title": title})
    return keyword


def generate_weekly_report(weekly_content_md: str, model="GPT-4-Turbo"):
    """Generate weekly report via LLMChain."""
    weekly_report_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ps.WEEKLY_SYSTEM_PROMPT),
        ("user", ps.WEEKLY_USER_PROMPT),
        (
            "user",
            "Tip: Remember to add plenty of citations! Use the format (arxiv:1234.5678)`.",
        ),
    ]
)
    weekly_report_chain = LLMChain(llm=llm_map[model], prompt=weekly_report_prompt)
    weekly_report = weekly_report_chain.run(weekly_content=weekly_content_md)
    return weekly_report


def write_tweet(previous_tweets:str, tweet_style: str, tweet_facts: str, model="GPT-4-Turbo"):
    """Write a tweet via LLMChain."""
    tweet_prompt = ChatPromptTemplate.from_messages([
        ("system", ps.TWEET_SYSTEM_PROMPT),
        ("user", ps.TWEET_USER_PROMPT),
    ])
    tweet_chain = LLMChain(llm=llm_map[model], prompt=tweet_prompt)
    tweet = tweet_chain.run(previous_tweets=previous_tweets, tweet_style=tweet_style, tweet_facts=tweet_facts)
    return tweet