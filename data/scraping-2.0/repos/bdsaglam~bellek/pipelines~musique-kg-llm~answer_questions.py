import json
from pathlib import Path

import typer
from dotenv import load_dotenv
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
from llama_index import ServiceContext, StorageContext, load_index_from_storage
from llama_index.callbacks import CallbackManager
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms import OpenAI
from rich.console import Console

from bellek.llama_index.obs import make_phoenix_trace_callback_handler
from bellek.utils import generate_time_id, set_seed

err = Console(stderr=True).print

load_dotenv()

set_seed(42)

set_llm_cache(SQLiteCache(database_path="/tmp/langchain-cache.db"))

# model to generate embeddings for triplets
embed_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")

# language model to use for triplet extraction
llm = OpenAI(temperature=0, model="gpt-3.5-turbo")

# Setup LLM observability
LLM_TRACES_FILEPATH = Path(f"/tmp/phoenix/thesis-kg-llm/qna/traces-{generate_time_id()}.jsonl")
callback_manager = CallbackManager(handlers=[make_phoenix_trace_callback_handler(LLM_TRACES_FILEPATH)])

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    callback_manager=callback_manager,
)


def load_index(directory: Path):
    storage_context = StorageContext.from_defaults(persist_dir=directory / "index")
    return load_index_from_storage(
        storage_context,
        service_context=service_context,
        include_embeddings=True,
    )


TEXT_QA_PROMPT_USER_MSG_CONTENT = """Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer in 2-4 words: """


def make_query_engine(index):
    query_engine = index.as_query_engine(
        include_text=True,
        embedding_mode="hybrid",
        response_mode="simple_summarize",
        verbose=True,
    )
    original_text_qa_prompt = query_engine.get_prompts()["response_synthesizer:text_qa_template"]
    original_text_qa_prompt.conditionals[0][1].message_templates[1].content = TEXT_QA_PROMPT_USER_MSG_CONTENT
    query_engine.update_prompts({"response_synthesizer:text_qa_template": original_text_qa_prompt})
    return query_engine


def answer_questions(query_engine, example):
    sub_questions = [item["question"] for item in example["question_decomposition"]]
    hop1_question = sub_questions[0]
    hop1_answer = query_engine.query(hop1_question).response
    example["question_decomposition"][0]["answer"] = hop1_answer
    hop2_question = sub_questions[1].replace("#1", hop1_answer)
    hop2_answer = query_engine.query(hop2_question).response
    example["question_decomposition"][1]["answer"] = hop2_answer
    return example


def main(
    dataset_file: Path = typer.Option(...),
    knowledge_graph_directory: Path = typer.Option(...),
    out: Path = typer.Option(...),
):
    with open(dataset_file) as src:
        with open(out, "w") as dst:
            for line in src:
                example = json.loads(line)
                id = example["id"]

                err(f"Answering the question in the sample {id}")
                query_engine = make_query_engine(load_index(knowledge_graph_directory / id))
                example_answered = answer_questions(query_engine, example)

                dst.write(json.dumps(example_answered, ensure_ascii=False))
                dst.write("\n")


if __name__ == "__main__":
    typer.run(main)
