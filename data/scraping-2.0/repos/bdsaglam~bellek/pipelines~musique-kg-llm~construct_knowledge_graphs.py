import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List

import kuzu
import typer
from dotenv import load_dotenv
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
from llama_index import Document, KnowledgeGraphIndex, ServiceContext
from llama_index.callbacks import CallbackManager
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.prompts.base import Prompt
from llama_index.prompts.prompt_type import PromptType
from llama_index.storage.storage_context import StorageContext
from phoenix.trace.llama_index import OpenInferenceTraceCallbackHandler
from phoenix.trace.schemas import Span
from pyvis.network import Network
from rich.console import Console

from bellek.llama_index.graph_stores.kuzu import KuzuGraphStore
from bellek.llama_index.obs import TraceRecorder
from bellek.utils import set_seed

err = Console(stderr=True).print

load_dotenv()

set_seed(42)

set_llm_cache(SQLiteCache(database_path="/tmp/langchain-cache.db"))


def make_service_context(llm_config: dict[str, Any], trace_callback: Callable[[List[Span]], None]):
    llm_type = llm_config["type"]
    llm_params = llm_config["params"]
    if llm_type == "llama2-sft":
        from bellek.llama_index.llms import HuggingFaceTextGenInferenceLLM

        llm = HuggingFaceTextGenInferenceLLM(inference_server_url="http://localhost:8080/", **llm_params)
    elif llm_type == "llama2-base":
        from llama_index.llms import Anyscale

        llm = Anyscale(api_key=os.getenv("ANYSCALE_API_KEY"), **llm_params)
    elif llm_type == "openai":
        from llama_index.llms import OpenAI

        llm = OpenAI(**llm_params)
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")

    # model to generate embeddings for triplets
    embed_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")

    # Setup LLM observability
    callback_manager = CallbackManager(handlers=[OpenInferenceTraceCallbackHandler(trace_callback)])

    return ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        # transformations=[],
        callback_manager=callback_manager,
    )


LLAMA2_KG_TRIPLET_EXTRACT_TMPL = """<s>[INST] <<SYS>>
You are a helpful assistant that extracts up to {max_knowledge_triplets}  entity-relation-entity triplets from given text. Use '|' as delimiter and provide one triplet per line. The entities in a triplet must be different.
<</SYS>>
Alaa Abdul Zahra plays for Al Shorta SC. His club is AL Kharaitiyat SC, which has its ground at, Al Khor. [/INST] Al Kharaitiyat SC|ground|Al Khor
Alaa Abdul-Zahra|club|Al Kharaitiyat SC
Alaa Abdul-Zahra|club|Al Shorta SC </s><s>[INST] {text} [/INST] """

DEFAULT_KG_TRIPLET_EXTRACT_TMPL = """
Some text is provided below. Given the text, extract up to {max_knowledge_triplets}  knowledge triplets in the form of (subject, predicate, object) that might be relevant to the following question. The subject and object must be different.
Prioritize triplets that:
1. Offer temporal information like 'founded in,' 'created on,' 'abolished in,' etc.
2. Provide spatial details such as 'located in,' 'borders,' 'from,' etc.
3. Show ownership or affiliation via terms like 'owned by,' 'affiliated with,' 'publisher of,' etc.
4. Offer identification or categorization like 'is,' 'are,' 'was,' etc.
Avoid stopwords.
---------------------
Example:
Question: When was the institute that owned The Collegian founded?
Text: The Collegian is the bi-weekly official student publication of Houston Baptist University in Houston, Texas.
Triplets:
(The Collegian, is, bi-weekly official student publication)
(The Collegian, owned by, Houston Baptist University)
(Houston Baptist University, in, Houston)
(Houston, in, Texas)
---------------------
Text: {text}
Triplets:
""".strip()


def make_erx_prompt(model_type: str):
    prompt_str = LLAMA2_KG_TRIPLET_EXTRACT_TMPL if "llama2" in model_type else DEFAULT_KG_TRIPLET_EXTRACT_TMPL
    return Prompt(
        prompt_str,
        prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT,
    )


def make_docs(example, only_supporting=False):
    ps = example["paragraphs"]
    for p in ps:
        if only_supporting and not p["is_supporting"]:
            continue
        idx = p["idx"]
        title = p["title"]
        body = p["paragraph_text"]
        is_supporting = p["is_supporting"]
        text = f"# {title}\n{body}"
        yield Document(
            text=text,
            metadata={"parent_id": example["id"], "idx": idx, "is_supporting": is_supporting},
            excluded_llm_metadata_keys=["parent_id", "idx", "is_supporting"],
        )


def visualize_knowledge_graph(index, out: Path):
    g = index.get_networkx_graph()
    net = Network(notebook=False, cdn_resources="in_line", directed=True)
    net.from_nx(g)
    net.save_graph(str(out))
    return out


def construct_knowledge_graph(
    example,
    *,
    max_triplets_per_chunk: int,
    include_embeddings: bool,
    llm_config: dict[str, Any],
    trace_callback: Callable[[List[Span]], None],
    out_dir: Path,
):
    id = example["id"]

    db = kuzu.Database(str(out_dir / "kuzu"))
    graph_store = KuzuGraphStore(db)
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    # Create documents to index into knowledge graph
    documents = list(make_docs(example, only_supporting=True))
    err(f"Created {len(documents)} documents for sample {id}")

    if "llama2" in llm_config["type"]:
        from bellek.jerx.utils import parse_triplets

        def _parse_triplet_response(response: str, max_length: int = 128) -> list[tuple[str, str, str]]:
            triplets = parse_triplets(response.strip())
            return [(e1, rel, e2) if e1!=e2 else (e1, rel, e2 + "(obj)") for e1,rel,e2 in triplets]

        KnowledgeGraphIndex._parse_triplet_response = staticmethod(_parse_triplet_response)

    # Create knowledge graph index
    err(f"Creating the knowledge graph index for sample {id}")
    index = KnowledgeGraphIndex.from_documents(
        documents=documents,
        max_triplets_per_chunk=max_triplets_per_chunk,
        storage_context=storage_context,
        service_context=make_service_context(llm_config, trace_callback),
        include_embeddings=include_embeddings,
        kg_triple_extract_template=make_erx_prompt(llm_config["type"]),
    )

    err(f"Persisting the knowledge graph index for sample {id}")
    index.storage_context.persist(persist_dir=(out_dir / "index"))

    err(f"Visualizing the knowledge graph for sample {id}")
    visualize_knowledge_graph(index, out_dir / "kuzu-network.html")


def main(
    dataset_file: Path = typer.Option(...),
    llm_config_file: Path = typer.Option(...),
    out: Path = typer.Option(...),
    ignore_errors: bool = typer.Option(False),
):
    llm_config = json.loads(llm_config_file.read_text())

    with open(dataset_file) as f:
        for line in f:
            example = json.loads(line)
            id = example["id"]

            example_out_dir = out / id
            shutil.rmtree(example_out_dir, ignore_errors=True)
            example_out_dir.mkdir(exist_ok=True, parents=True)

            try:
                err(f"Constructing the knowledge graph for the sample {id}")
                trace_callback = TraceRecorder((example_out_dir / "traces.jsonl").open("w"))
                max_triplets_per_chunk = random.randint(4, 8)
                construct_knowledge_graph(
                    example,
                    max_triplets_per_chunk=max_triplets_per_chunk,
                    include_embeddings=False,
                    llm_config=llm_config,
                    trace_callback=trace_callback,
                    out_dir=example_out_dir,
                )
            except Exception as exc:
                err(f"Failed to construct the knowledge graph for sample {id}.\n{exc}")
                if not ignore_errors:
                    raise exc

    (out / "timestamp.txt").write_text(str(datetime.now().isoformat(timespec="milliseconds")))


if __name__ == "__main__":
    typer.run(main)
