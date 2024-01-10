from llama_index import GPTVectorStoreIndex, GPTListIndex
from llama_index.indices.composability import ComposableGraph
from llama_index import StorageContext, load_index_from_storage
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index import GPTTreeIndex, GPTVectorStoreIndex
from llama_index.tools import QueryEngineTool
from llama_index.query_engine import RouterQueryEngine


from key1 import KEY
import os
import openai

os.environ['OPENAI_API_KEY'] = KEY

def generate_response32(prompt,dir_list1):
    print (dir_list1)
    summ_list = ["summary"+ str(each) for each in range(len(dir_list1)) ]
    index_list1 = []
    for dir1 in dir_list1:
        print("diresss", dir1)
        v_dir = f'./vector/{dir1}'
        print ("dirs", v_dir)
        storage_context = StorageContext.from_defaults(persist_dir=v_dir)
        index1 = load_index_from_storage(storage_context)
        print ("NOW",index1)
        tool1 = QueryEngineTool.from_defaults(
                query_engine=index1.as_query_engine(),
                description=dir1,
                )

        index_list1.append(tool1)
    #print ("index", index, dir(index))

    query_engine = RouterQueryEngine.from_defaults(
    query_engine_tools=index_list1 )
    response = query_engine.query(  prompt  )
    print (response, type(response))
    return str(response)


def generate_response45(prompt,dir_list1):
    print (dir_list1)
    summ_list = ["summary"+ str(each) for each in range(len(dir_list1)) ]
    index_list1 = []
    for dir1 in dir_list1:
        print("diresss", dir1)
        storage_context = StorageContext.from_defaults(persist_dir=f'./vector/{dir1}')
        index1 = load_index_from_storage(storage_context)
        index_list1.append(index1)
    print (index_list1)
    graph = ComposableGraph.from_indices(GPTTreeIndex, index_list1, index_summaries=summ_list)
    #graph.save_to_disk("graph.json")
    print("dir on graph", dir(graph) )
    # set query config
    query_configs = [
        {
            "index_struct_type": "simple_dict",
            "query_mode": "default",
            "query_kwargs": {
                "similarity_top_k": 3,
                "response_mode": "tree_summarize"
            }
        },
    ]

    #graph.save_to_disk("./vector")
    print ("all_indices", graph.all_indices)
    #index = ComposableGraph.load_from_disk("graph.json")
    index = graph.as_query_engine()
    response=index.query(prompt)
    print (dir(index))

    print (response, type(response))
    return str(response)


def generate_response121(prompt,dir_list1):
    print (dir_list1)
    summ_list = [each for each in dir_list1 ]
    index_list1 = []
    for dir1 in dir_list1:
        print("diresss", dir1)
        storage_context = StorageContext.from_defaults(persist_dir=f'./vector/{dir1}')
        index1 = load_index_from_storage(storage_context)
        index_list1.append(index1)
    print (index_list1)
    graph = ComposableGraph.from_indices(GPTTreeIndex, index_list1, index_summaries=summ_list)

    # configure query engines
    custom_query_engines = {
        index_list1[0].index_id: index_list1[0].as_query_engine(
            similarity_top_k=3,
            response_mode="tree_summarize",
        )
    }

    query_engine = graph.as_query_engine(
    custom_query_engines=custom_query_engines
    )


    response = query_engine.query(
        prompt,
    )


    print (response, type(response))
    return str(response)


def generate_response(prompt,dir_list1):
    print (dir_list1)
    summ_list = []
    index_list1 = []

    for dir1 in dir_list1:
        v_dir = f'./vector/{dir1}'
        print("diresss", dir1)
        storage_context = StorageContext.from_defaults(persist_dir=v_dir)
        index1 = load_index_from_storage(storage_context)
        index_list1.append(index1)
        summ_list.append(dir1)

    graph = ComposableGraph.from_indices(
    GPTListIndex,
    index_list1,
    index_summaries=summ_list,
    storage_context=StorageContext.from_defaults(),
    )
    '''
    print ("index*********", index_list1)
    storage_context1 = StorageContext.from_defaults()

    storage_context1.persist(persist_dir=f"./vector/all")
    storage_context2 = StorageContext.from_defaults(persist_dir=f"./vector/all")
    index1 = load_index_from_storage(storage_context2)
    query_engin = index1.as_query_engine()
    response = query_engin.query(prompt)
    print (response, type(response))
    return str(response)
    '''
    query_engine = graph.as_query_engine( 
    )


    response = query_engine.query(
        prompt,
    )


    print (response, type(response))
    return str(response)

def generate_response80(prompt,dir_list1):
    from llama_index.indices.list.base import GPTListIndex
    print (dir_list1)
    summ_list = []
    index_list1 = []

    for dir1 in dir_list1:
        v_dir = f'./vector/{dir1}'
        print("diresss", dir1)
        storage_context = StorageContext.from_defaults(persist_dir=v_dir)
        index1 = load_index_from_storage(storage_context)
        print ()
        print (dir(index1))
        #print ("strogae***********", dir(storage_context))
        index_list1.append(index1)
        summ_list.append(dir1)

    index = GPTListIndex(index_list1)
    response=index.query(prompt)
    print (dir(index))

    print (response, type(response))
    return str(response)
