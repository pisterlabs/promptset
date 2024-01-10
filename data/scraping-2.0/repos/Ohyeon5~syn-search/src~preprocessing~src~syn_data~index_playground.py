import argparse
import os
from pathlib import Path

# from langchain import hub
# from llama_index.prompts import LangchainPromptTemplate
from dotenv import load_dotenv
from llama_index import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    set_global_service_context,
)
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index.indices.composability import ComposableGraph
from llama_index.llms import AzureOpenAI
from llama_index.prompts import PromptTemplate
from syn_data.index_builder import build_index_per_file
from syn_data.path import MODULE_PATH

# In this script, explore different index loaders
# 1. JSON/XML loader
# 2. combine multiple indices
# 3. Test with examples


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine multiple indices",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build index",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    data_root_dir = MODULE_PATH / ".." / ".." / ".." / ".." / "data"
    data_dir = data_root_dir / "uspto_json" / "grants"

    if args.build_index:
        val_set = [
            data_dir / "I20160906.json",
            data_dir / "I20160913.json",
            data_dir / "I20160920.json",
        ]

        save_dir = data_root_dir / "index"
        save_dir.mkdir(exist_ok=True)

        # Build index per file
        for json_file in val_set:
            build_index_per_file(
                json_file,
                save_dir / f"{str(json_file.name).split('.')[0]}.index",
            )

    # query from the index
    input_text = "[Cl:1][C:2]1[CH:3]=[C:4]([CH3:29])[C:5]2[N:10]=[C:9]([C:11]3[N:15]([C:16]4[C:21]([Cl:22])=[CH:20][CH:19]=[CH:18][N:17]=4)\
[N:14]=[C:13]([C:23]([F:26])([F:25])[F:24])[CH:12]=3)[O:8][C:7](=[O:27])[C:6]=2[CH:28]=1.[CH3:30][NH2:31]>O1CCCC1>[Cl:1][C:2]1[CH:28]=\
[C:6]([C:7]([NH:31][CH3:30])=[O:27])[C:5]([NH:10][C:9]([C:11]2[N:15]([C:16]3[C:21]([Cl:22])=[CH:20][CH:19]=[CH:18][N:17]=3)[N:14]=\
[C:13]([C:23]([F:26])([F:24])[F:25])[CH:12]=2)=[O:8])=[C:4]([CH3:29])[CH:3]=1"
    # # Combine indices
    load_dotenv()
    llm = AzureOpenAI(
        engine="gpt-35-turbo",
        model="gpt-35-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("OPENAI_BASE_URL"),
    )

    embed_model = AzureOpenAIEmbedding(
        azure_deployment="text-embedding-ada-002",
        model="text-embedding-ada-002",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("OPENAI_BASE_URL"),
        embed_batch_size=1,
    )

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )

    set_global_service_context(service_context)

    if args.combine:
        indices = []
        index_summaries = []
        for index_path in Path("data/index").glob("*.index"):
            # Load vector db
            print(f"loading index from {index_path}")
            storage_context = StorageContext.from_defaults(persist_dir=index_path)
            index = load_index_from_storage(storage_context)
            index_summary = (
                index.as_query_engine().query("reaction smiles of this node").response
            )
            indices.append(index)
            index_summaries.append(index_summary)

        patent_index = ComposableGraph.from_indices(
            VectorStoreIndex,
            indices,
            index_summaries=index_summaries,
            service_context=service_context,
        )
    else:
        index_path = Path("data/index/I20160906.index")
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        patent_index = load_index_from_storage(storage_context)

    # # rag prompt
    # langchain_prompt = hub.pull("rlm/rag-prompt")
    # lc_prompt_tmpl = LangchainPromptTemplate(
    #     template=langchain_prompt,
    #     template_var_mappings={"query_str": "question", "context_str": "context"},
    # )

    prompt = "You are to describe an experimental procedure for an organic chemistry \
         reaction based on an input reaction smiles. This should take the form of a formal experimental, \
         as would be found in a publication. Followed by a simplified format of the reaction. \
        Then provide a list of possible safety concerns. You should use the context provided to suggest\
        the best response\nYou are forbidden from showing me smiles strings. "

    # chemist assistant prompt
    new_summary_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "You are to describe an experimental procedure for an organic chemistry \
         reaction based on an input reaction smiles. This should take the form of a formal experimental, \
         as would be found in a publication. Followed by a simplified format of the reaction. \
        Then provide a list of possible safety concerns. You should use the context provided to suggest\
        the best response\nYou are forbidden from showing me smiles strings. \n"
        "If you are unable to answer the question, please respond with 'I don't know'.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)
    query_engine = patent_index.as_query_engine(similarity_top_k=5)
    # query_engine.update_prompts(
    #     {"response_synthesizer:text_qa_template": lc_prompt_tmpl}
    # )
    query_engine.update_prompts(
        {"response_synthesizer:summary_template": new_summary_tmpl}
    )
    response = query_engine.query(prompt + input_text)
    print(response.response)
