import shutil
import subprocess
from pathlib import Path
from typing import List
import os
import pandas as pd
import re
import markdownify
from dagster import AssetExecutionContext, FreshnessPolicy, MetadataValue, RetryPolicy, asset
from langchain.cache import SQLiteCache
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.globals import set_llm_cache
from langchain.memory import ConversationSummaryMemory
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent


from langchain_surreal_db_integration import SurrealDB as SurrealDBVectorStore


set_llm_cache(SQLiteCache(database_path=".langchain.db"))


ARTIFACTS_PATH = Path("./artifacts/")
INDEX_PATH = ARTIFACTS_PATH / "faiss_search_index.pickle"
URL_WITH_DOCS = "https://surrealdb.com/docs/"
MODEL_NAME = "gpt-4"

def get_docs(md_folder: Path):
    docs = []
    for md_file in md_folder.iterdir():
        with open(md_file, "r") as f:
            url = URL_WITH_DOCS + str(md_file.stem).replace(".", "/")
            d = Document(page_content=f.read(), metadata={"source": url})
        docs.append(d)
    return docs


def hbs_to_markdown(hbs_path, md_path, snippets_folder):
    with open(hbs_path, "r") as f:
        hbs_content = f.read()

    # # Embed external snippets
    def embed_snippets_code_from_file(match):
        snippet_name = match.group(1)
        with open(os.path.join(snippets_folder, snippet_name), "r") as snippet_file:
            return f"```\n{snippet_file.read()}\n```"

    def embed_snippets_code_inline(match):
        snippet_name = match.group(1)
        snippet_path = os.path.join(snippets_folder, snippet_name)
        # print(f"snippet_path = {snippet_path}")
        # Check if the file exists
        if os.path.exists(snippet_path):
            with open(snippet_path, "r") as snippet_file:
                return f"```\n{snippet_file.read()}\n```"
        else:
            # If the file does not exist, use the inline code
            inline_code = match.group(2) if match.group(2) else ""
            return f"```\n{inline_code}\n```"

    hbs_content = re.sub(r'<Code @name="([^"]+)" [^>]*>([\s\S]*?)<\/Code>', embed_snippets_code_inline, hbs_content)
    hbs_content = re.sub(r'<Code @name="([^"]+)" [^>]*>', embed_snippets_code_from_file, hbs_content)
    hbs_content = re.sub(r"\{\{.*?\}\}", "", hbs_content)
    markdown_content = markdownify.markdownify(hbs_content, heading_style="ATX")

    with open(md_path, "w") as f:
        f.write(markdown_content)

def convert_all_docs(
    md_folder: Path, hbs_root: Path, snippets_folder: str
):
    md_folder.mkdir(exist_ok=True, parents=True)

    for root, _, files in os.walk(hbs_root):
        for file in files:
            if file.endswith(".hbs"):
                full_path = Path(root) / file
                with open(full_path, "r") as f:
                    relative_path = full_path.relative_to(hbs_root)
                    key = ".".join(relative_path.parts[:-1]) + "." + file.rsplit(".", 1)[0]

                    hbs_to_markdown(
                        hbs_path=full_path, md_path=md_folder / f"{key}.md", snippets_folder=snippets_folder
                    )

def get_github_docs(repo_owner: str, repo_name: str) -> Path:
    d = ARTIFACTS_PATH
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    subprocess.check_call(f"git clone --depth 1 https://github.com/{repo_owner}/{repo_name}.git", cwd=d, shell=True)
    return d / repo_name


@asset
def www_surrealdb_com_repo(context: AssetExecutionContext):
    context.add_output_metadata(
        metadata={
            "surrealdb_docs_url": MetadataValue.url("https://surrealdb.com/docs"),
        }
    )

    return get_github_docs("surrealdb", "www.surrealdb.com")


@asset
def surreal_markdown_docs(context: AssetExecutionContext, www_surrealdb_com_repo: Path):
    md_folder = Path("./artifacts/docs")
    convert_all_docs(
        md_folder=md_folder,
        hbs_root=www_surrealdb_com_repo / "app/templates/docs",
        snippets_folder=www_surrealdb_com_repo / "app/snippets/",
    )

    with open(md_folder / 'introduction.mongo.md', 'r') as f:
        md_str = f.read()

    context.add_output_metadata(
        metadata={
            'Details': MetadataValue.md(md_str)
        }
    )

    return md_folder


@asset
def surreal_langchain_docs(surreal_markdown_docs: Path) -> List[Document]:
    return get_docs(md_folder=surreal_markdown_docs)


@asset(
    retry_policy=RetryPolicy(max_retries=5, delay=5),
    freshness_policy=FreshnessPolicy(maximum_lag_minutes=60 * 24),
)
def surreal_db_search_index(surreal_langchain_docs):
    markdown_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN, chunk_size=2000, chunk_overlap=200)
    texts = markdown_splitter.split_documents(surreal_langchain_docs)

    index = SurrealDBVectorStore.from_documents(texts, OpenAIEmbeddings())
    return index


@asset
def list_of_test_cases(context: AssetExecutionContext):
    test_cases = [
        "What is SurrealDB?",
        "How can I deploy SurrealDB on AWS?",
        "How can I get the schema of my database?",
        "Write an SQL query to select 5 rows from a table.",
        "Can you provide an example of a live query?",
        "Why do I need a live query?",
        "How could upload banch of data into SurrealDB? Could I use csv, tsv, json or parquet format?",
        "Does SurrealDB have demo data?",
        "How does surreal import command work?",
        "Could you some me all CREATE statments for SurrealDB?",
        "How could I convert this SQL code 'SELECT COUNT(*) FROM hits;' into Surreal Query Language?"
        "How could I convert this SQL code 'SELECT AdvEngineID, COUNT(*) FROM hits WHERE AdvEngineID <> 0 GROUP BY AdvEngineID ORDER BY COUNT(*) DESC;' into Surreal Query Language?"
    ]
    context.add_output_metadata(
        metadata={
            "test_cases": test_cases,
        }
    )
    return test_cases


@asset
def qa_chain_examples(context: AssetExecutionContext, surreal_db_search_index, list_of_test_cases):
    test_cases = []

    llm = ChatOpenAI(model_name=MODEL_NAME, temperature = 0)
    chain = load_qa_with_sources_chain(llm)

    for q in list_of_test_cases:
        r = chain(
            {
                "input_documents": surreal_db_search_index.similarity_search(q, k=16),
                "question": q,
            },
            return_only_outputs=True,
        )["output_text"]
        test_cases.append({"q": q, "r": r})
    df = pd.DataFrame(test_cases)
    context.add_output_metadata(
        metadata={
            "num_cases": len(df),
            "preview": MetadataValue.md(df.to_markdown()),
        }
    )
    return df


@asset
def conv_retrieval_examples(context: AssetExecutionContext, surreal_db_search_index, list_of_test_cases):
    retriever = surreal_db_search_index.as_retriever(search_kwargs={'fetch_k': 16})

    llm = ChatOpenAI(model_name=MODEL_NAME, temperature = 0)
    memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

    test_cases = []
    for q in list_of_test_cases:
        result = qa(q)
        r = result["answer"]
        test_cases.append({"q": q, "r": r})
    df = pd.DataFrame(test_cases)
    context.add_output_metadata(
        metadata={
            "num_cases": len(df),
            "preview": MetadataValue.md(df.to_markdown()),
        }
    )
    return df

@asset
def agent_retrieval_examples(context: AssetExecutionContext, surreal_db_search_index, list_of_test_cases):
    retriever = surreal_db_search_index.as_retriever(search_kwargs={'fetch_k': 16})
    tool = create_retriever_tool(retriever, "surreal-db-docs", "Searches and returns documents about SurreadDB.")
    tools = [tool]

    
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature = 0)
    agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)


    test_cases = []
    for q in list_of_test_cases:
        result = agent_executor({"input": q})
        r = result["output"]
        test_cases.append({"q": q, "r": r})
    df = pd.DataFrame(test_cases)
    context.add_output_metadata(
        metadata={
            "num_cases": len(df),
            "preview": MetadataValue.md(df.to_markdown()),
        }
    )
    return df


@asset
def all_test_cases(context: AssetExecutionContext, agent_retrieval_examples, conv_retrieval_examples, qa_chain_examples):
    all_test_cases = pd.DataFrame({
        'q': agent_retrieval_examples['q'],

        'qa_chain': qa_chain_examples['r'],
        'conv_retrieval': conv_retrieval_examples['r'],
        'agent_retrieval': agent_retrieval_examples['r'],

    })
    context.add_output_metadata(
        metadata={
            "num_cases": len(all_test_cases),
            "preview": MetadataValue.md(all_test_cases.to_markdown()),
        }
    )
    return all_test_cases    
