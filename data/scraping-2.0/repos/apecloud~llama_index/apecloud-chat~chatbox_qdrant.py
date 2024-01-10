import sys
import os
import argparse
import logging
from langchain import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from gpt_index.langchain_helpers.agents import LlamaToolkit, create_llama_agent, create_llama_chat_agent, IndexToolConfig, GraphToolConfig
from gpt_index.indices.query.query_transform.base import DecomposeQueryTransform
from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader, QuestionAnswerPrompt
from gpt_index import LLMPredictor, PromptHelper, ServiceContext
from gpt_index.indices.composability import ComposableGraph
from gpt_index.indices.base import IS
from gpt_index import GPTListIndex, SimpleDirectoryReader
import qdrant_client
from gpt_index.vector_stores.qdrant import QdrantVectorStore
from gpt_index.data_structs.data_structs_v2 import V2IndexStruct,IndexDict
from gpt_index import GPTQdrantIndex
from read_key import read_key_from_file

def parse_arguments():
    parser = argparse.ArgumentParser(description="Query Engine for KubeBlocks.")
    parser.add_argument("key_file", type=str, help="Key file for OpenAI_API_KEY.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    key_file = args.key_file

    openai_api_key = read_key_from_file(key_file)
    # set env for OpenAI api key
    os.environ['OPENAI_API_KEY'] = openai_api_key
    print(f"OPENAI_API_KEY:{openai_api_key}")

    # set log level
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))# define LLM

    # initialize the qdrant client
    client = qdrant_client.QdrantClient(
        url="http://localhost",
        port=6333,
        grpc_port=6334,
        prefer_grpc=False,
        https=False,
        timeout=300
    )

    # initialize the index set with codes and documents
    index_set = {}
    index_struct = IndexDict(summary=None)

    index_set["Docs"] = GPTQdrantIndex(client=client, collection_name="kubeblocks_doc", index_struct=index_struct)
    index_set["Code"] = GPTQdrantIndex(client=client, collection_name="kubeblocks_code", index_struct=index_struct)
    index_set["Config"] = GPTQdrantIndex(client=client, collection_name="kubeblocks_config", index_struct=index_struct)

    # initialize summary for each index
    index_summaries = ["design and user documents for kubeblocks", "codes of implementations of kubeblocks", "config for kubeblocks"]

    # define a LLMPredictor
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

    # define prompt helper
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_output = 4096
    # set maximum chunk overlap
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    # define a list index over the vector indices
    # allow us to synthesize information across each index
    graph = ComposableGraph.from_indices(
        root_index_cls = GPTListIndex,
        children_indices = [index_set["Docs"], index_set["Code"], index_set["Config"]],
        index_summaries = index_summaries,
        service_context = service_context,
    )

    decompose_transform = DecomposeQueryTransform(
        llm_predictor, verbose=True
    )

    query_configs = [
        {
            "index_struct_type": "simple_dict",
            "query_mode": "default",
            "query_kwargs":{
                "similarity_top_k": 3,
            },
            "query_transform": decompose_transform
        },
        {
            "index_struct_type": "list",
            "query_mode": "default",
            "query_kwargs": {
                "response_mode": "tree_summarize",
                "verbose": True
            }
        },
    ]

    # graph config
    graph_config = GraphToolConfig(
        graph = graph,
        name = f"Graph Index",
        description = "useful when you want to answer queries that about how to use and develop with kubeblocks",
        query_configs = query_configs,
        tool_kwargs = {"return_direct": True, "return_sources": True},
        return_sources=True
    )

    # define toolkit
    index_configs = []
    tool_config = IndexToolConfig(
        index = index_set["Docs"],
        name = f"Vector Index Docs",
        description = '''answer questions about how to install, deploy, maintain kubeblocks; \ 
        questions about clusterdefinition, clusterversion, cluster; \
        qusetions about lifecycle, monitoring, backup, safety for all kinds of atabases''',
        index_query_kwargs = {"similarity_top_k": 3},
        tool_kwargs = {"return_direct": True, "return_sources": True}
    )
    index_configs.append(tool_config)

    tool_config = IndexToolConfig(
        index = index_set["Code"],
        name = f"Vector Index Code",
        description = '''answer questions about the code implementations of kubeblocks; 
        questions about the code of clusterdefinition, clusterversion, cluster;
        qusetions about lifecycle, monitoring, backup, safety for all kinds of atabases''',
        index_query_kwargs = {"similarity_top_k": 3},
        tool_kwargs = {"return_direct": True, "return_sources": True}
    )
    index_configs.append(tool_config)

    tool_config = IndexToolConfig(
        index = index_set["Config"],
        name = f"Vector Index Config",
        description = '''answer questions about the generation of configs in kubeblocks; 
        questions about the configs of clusterdefinition, clusterversion, cluster;
        backuppolicy, backup, RBAC, OpsRequest, podSpec, containers, volumeClaimTemplates, volumes''',
        index_query_kwargs = {"similarity_top_k": 3},
        tool_kwargs = {"return_direct": True, "return_sources": True}
    )
    index_configs.append(tool_config)

    tookit = LlamaToolkit(
        index_configs = index_configs,
        graph_configs = [graph_config]
    )

    # create the llama agent
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = OpenAI(temperature=0)
    agent_chain = create_llama_agent(
        tookit,
        llm,
        #agent = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory = memory,
        verbose = True
    )

    while True:
        text_input = input("User:")
        response = agent_chain.run(input=text_input)
        print(f"Agent: {response}")


if __name__ == "__main__":
    main()

