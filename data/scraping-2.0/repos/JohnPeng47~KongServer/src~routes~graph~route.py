from pydantic import BaseModel
from fastapi import APIRouter

from .schema import GraphMetadataResp, GraphNode, RFNode, SaveGraphReq, rfnode_to_kgnode
from .db import get_graph_metadata_db, get_graph_db, delete_graph_db, delete_graph_metadata_db 

from fastapi.requests import Request
import uuid

from .utils import merge_tree_ids

from KongBot.bot.base import KnowledgeGraph
from KongBot.bot.explorationv2.llm import GenSubTreeQuery, Tree2FlatJSONQuery, GenSubTreeQueryV2
from KongBot.bot.adapters.ascii_tree_to_kg import ascii_tree_to_kg

from typing import List
import json

from langchain import LLMChain, PromptTemplate, OpenAI

router = APIRouter()


@router.get("/metadata/", response_model=List[GraphMetadataResp])
def get_graph_metadata():
    metadata_list = []
    # consider returning a cursor here to be more memory efficient
    # although the pagination limit should do the trick?
    metadatas = get_graph_metadata_db(pagination=6)
    for document in metadatas:
        metadata_list.append(document)
    return metadata_list

@router.get("/graph/{graph_id}", response_model=GraphNode)
def get_graph(graph_id: str, request: Request):
    if not request.app.curr_graph or request.app.curr_graph != graph_id:
        request.app.curr_graph = KnowledgeGraph.load_graph(graph_id)

    kg: KnowledgeGraph = request.app.curr_graph
    
    return json.loads(kg.to_json_frontend())

@router.post("/graph/update")
def update_graph(rf_graph: RFNode, request: Request):
    import random    
    kg: KnowledgeGraph = request.app.curr_graph
    
    test_file = f"test_delete{random.randint(0,155)}.json"
    print("Writing to: ", test_file)
    with open(test_file, "w") as test:
        test.write(json.dumps(rf_graph.dict(), indent=4))

    kg_graph = rfnode_to_kgnode(rf_graph)
    kg.add_node(kg_graph, merge=True)
    print(kg.display_tree())

@router.get("/graph/delete/{graph_id}")
def delete_graph(graph_id: str, request: Request):
    delete_graph_db(graph_id)
    delete_graph_metadata_db(graph_id)

@router.post("/graph/save")
def update_graph(save_req: SaveGraphReq, request: Request):
    rf_graph, title = save_req.graph, save_req.title

    kg_graph = rfnode_to_kgnode(rf_graph)
    # assign different ID so it gets saved as a new graph
    kg_graph["id"] = str(uuid.uuid4())

    new_kg: KnowledgeGraph = KnowledgeGraph("test")
    new_kg.from_json(kg_graph)
    new_kg.save_graph(title = title)

@router.post("/gen/subgraph/", response_model=GraphNode)
def gen_subgraph(rf_subgraph: RFNode, request: Request):
    kg: KnowledgeGraph = request.app.curr_graph

    rf_subgraph_json = rfnode_to_kgnode(rf_subgraph)

    kg.add_node(rf_subgraph_json, merge=True)
    subtree = kg.display_tree_v2_lineage(rf_subgraph_json["id"])

    print("Subtree to generate: ", subtree)

    # GENERATE SUBTREE
    subtree = GenSubTreeQueryV2(kg.curriculum,
                                subtree,
                                model="gpt3").get_llm_output()
    subtree_node_new = ascii_tree_to_kg(subtree, rf_subgraph_json)
    kg.add_node(subtree_node_new, merge=True)

    print("New subtree: ", kg.display_tree(rf_subgraph_json["id"]))

    ## TODO: consider just returning the subgraph
    return json.loads(kg.to_json_frontend())