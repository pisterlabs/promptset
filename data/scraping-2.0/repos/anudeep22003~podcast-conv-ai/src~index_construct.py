from llama_index import Document
import json, os
from llama_index.node_parser import SimpleNodeParser
from llama_index import GPTTreeIndex, LLMPredictor, PromptHelper, GPTListIndex
from langchain import OpenAI
from llama_index.composability import ComposableGraph

from llama_index.data_structs.node_v2 import Node, DocumentRelationship


class ConfigLLM:
    # define LLM
    name = "gpt-3.5-turbo"
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo"))

    # define prompt helper
    # set maximum input size
    max_input_size = 2096
    # set number of output tokens
    num_output = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)


def index_construct_and_save(timechunk_path: str, save_loc: str):
    for filename in os.listdir(timechunk_path):
        file = os.path.join(timechunk_path, filename)
        data = json.load(open(file=file, mode="r"))
        # keys, text = list(zip(*data.items()))
        nodes = [Node(text=text, doc_id=keys) for keys, text in data.items()]
        index = GPTTreeIndex(nodes=nodes)
        index.save_to_disk(f"{save_loc}/{filename}.json")


def load_index_with_summary(index_loc: str):
    index_list = []
    index_summary_list = []
    for filename in os.listdir(index_loc):
        index_file = os.path.join(index_loc, filename)
        index = GPTTreeIndex.load_from_disk(index_file)
        summary = index.query(
            "What is the summary of this document chunk?", mode="summarize"
        )
        index_summary_list.append(str(summary))
        index_list.append(index)
        #! logging

    print("index list", len(index_list), index_list)

    return index_list, index_summary_list


def compose_graph_and_save(index_loc: str, save_loc: str):
    index_list, index_summary_list = load_index_with_summary(index_loc)
    #! logging
    print(index_summary_list)
    graph = ComposableGraph.from_indices(GPTListIndex, index_list, index_summary_list)
    graph.save_to_disk(save_loc)


def load_graph(graph_location: str):
    return ComposableGraph.load_from_disk(graph_location)


def query_graph(query: str, graph: ComposableGraph):
    response = graph.query(query, query_configs=get_query_configs())
    return response


def parse_response(response: ComposableGraph.query):
    print("-" * 50)
    print(response)
    print("-" * 50)
    print(
        str(response),
        # response.source_nodes,
        [node_with_score.node.doc_id for node_with_score in response.source_nodes],
        # [node.ref_doc_id for node in response.source_nodes],
        response.get_formatted_sources(),
        sep="\n" + "+" * 80 + "\n",
    )
    print("-" * 50)


def query_composed_index(query: str, graph_loc: str):
    graph = load_graph(graph_loc)
    response = query_graph(query, graph)
    parse_response(response)


def query_single_index(query: str, index_loc: str):
    index = GPTTreeIndex.load_from_disk(index_loc)
    response = index.query(query)
    parse_response(response)


def get_query_configs():
    # set query config
    query_configs = [
        {
            "index_struct_type": "simple_dict",
            "query_mode": "default",
            "query_kwargs": {"similarity_top_k": 1},
        },
        {
            "index_struct_type": "keyword_table",
            "query_mode": "simple",
            "query_kwargs": {},
        },
    ]
    return query_configs
