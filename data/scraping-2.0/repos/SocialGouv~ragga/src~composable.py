
from index import results
#from langchain import OpenAI
#import openai

from llama_index.indices.composability import ComposableGraph
from llama_index.indices.tree import GPTTreeIndex
from llama_index import ListIndex, VectorStoreIndex, GPTVectorStoreIndex,LLMPredictor,ServiceContext,StorageContext
from llama_index.query_engine.transform_query_engine import \
    TransformQueryEngine
from llama_index.indices.query.query_transform.base import \
    DecomposeQueryTransform
from llama_index.langchain_helpers.agents import (IndexToolConfig,
                                                  LlamaToolkit,
                                                  create_llama_chat_agent)
#from llama_index.storage.storage_context import StorageContextSingleton

# indices = results

# graph = ComposableGraph.from_indices(
#     SummaryIndex,
#     [index1, index2, index3],
#     index_summaries=[index1_summary, index2_summary, index3_summary],
#     storage_context=storage_context,
# )

#print(results)

#print(results[0][1])

all_indices = list(map(lambda a: a[0], results))
all_indices_summmaries = list(map(lambda a: a[1].get("description"), results))






graph = ComposableGraph.from_indices(
    root_index_cls=VectorStoreIndex,
    children_indices=all_indices,
    index_summaries=all_indices_summmaries,
    storage_context=StorageContext.from_defaults(),
)
query_engine = graph.as_query_engine()


query = "c'est quoi domifa ?"

res = query_engine.query(query)

# res = agent_chain.run({"chat_history":[], "input": query})
# # print(all_indices_summmaries)
# # graph = ComposableGraph.from_indices(ListIndex, children_indices=all_indices, index_summaries=all_indices_summmaries)

# # #print(graph)

# # query_engine = graph.as_query_engine(similarity_top_k=5)

# # query = "c'est quoi domifa ?"
# # res = query_engine.query(query)

print("---")
print(res)

print(dir(res))
print("---")
#print(dir(agent_chain))
print("---")
print(query)
print("---")
print(res)
print("---")
print(res.get_formatted_sources())




# llm_predictor = LLMPredictor()
# service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# graph = ComposableGraph.from_indices(
#     ListIndex,
#     all_indices,
#     index_summaries=all_indices_summmaries,
#     service_context=service_context
# )


# decompose_transform = DecomposeQueryTransform(
#     llm_predictor, verbose=True
# )

# custom_query_engines = {}
# for index in all_indices:
#     query_engine = index.as_query_engine()
#     query_engine = TransformQueryEngine(
#         query_engine,
#         query_transform=decompose_transform,
#         transform_metadata={'index_summary': index.index_struct.summary},
#     )
#     custom_query_engines[index.index_id] = query_engine
# custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(
#     response_mode='tree_summarize',
#     verbose=True,
# )

# graph_query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)

# def to_tool_config(result):
#     index, source = result
#     return IndexToolConfig(
#         query_engine=index.as_query_engine(similarity_top_k=3,),
#         name=source.get("title"),
#         description=source.get("description"),
#         tool_kwargs={"return_direct": True, "return_sources": True},
#     )

# tool_configs= list(map(to_tool_config, results))


# print(tool_configs)

# graph_config = IndexToolConfig(
#     query_engine=graph_query_engine,
#     name=f"Graph Index",
#     description="Pour toutes les questions liées à la fabrique des ministères sociaux ou à betagouv.",
#     tool_kwargs={"return_direct": True, "return_sources": True},
#     return_sources=True
# )

# toolkit = LlamaToolkit(
#     index_configs=tool_configs,
#    # graph_configs=[graph_config]
# )

# llm=OpenAI(temperature=0)

# agent_chain = create_llama_chat_agent(
#     toolkit,
#     llm=llm
# )



# query = "c'est quoi domifa ?"

# res = agent_chain.run({"chat_history":[], "input": query})
# # print(all_indices_summmaries)
# # graph = ComposableGraph.from_indices(ListIndex, children_indices=all_indices, index_summaries=all_indices_summmaries)

# # #print(graph)

# # query_engine = graph.as_query_engine(similarity_top_k=5)

# # query = "c'est quoi domifa ?"
# # res = query_engine.query(query)

# print("---")
# print(dir(res))
# print("---")
# print(dir(agent_chain))
# print("---")
# print(query)
# print("---")
# print(res)
# print("---")
# #print(res.get_formatted_sources())