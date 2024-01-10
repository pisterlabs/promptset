from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain.chains.llm import LLMChain
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from py2neo import Graph
import numpy as np
from CustomLibrary.Graph_Queries import (
    find_shortest_paths, 
    query_direct, 
    query_between_direct,
    get_node_labels_dict,
    get_node_label
)
from CustomLibrary.Graph_Utils import (
    select_paths2, 
    generate_answer, 
)
import gc
from CustomLibrary.OPC_Utils import extract_nodes_and_relationships

class KnowledgeGraphRetrieval:
    def __init__(self, uri, username, password, llm, entity_types, additional_entity_types=None):
        self.graph = Graph(uri, auth=(username, password))
        self.llm = llm
        self.entity_types = entity_types
        self.additional_entity_types = additional_entity_types  # Store the additional entity types dictionary

    def _call(self, names_list, question, progress_callback=None):

        entities_list = list(self.entity_types.items())

        # The first entity is the source entity
        source_entity_name, source_entity_type = entities_list[0]

        # The second entity is the target entity
        target_entity_name, target_entity_type = entities_list[1]

        result = find_shortest_paths(self.graph, 
                                        names_list, entities_list)

        source_to_target_paths = []
        selected_source_to_target_paths = []
        selected_source_to_target_graph_rels = []

        if result is None and source_entity_type == "Drug" or "Food" or "Metabolite":
            (selected_from_source_paths, 
            selected_from_source_nodes, 
            selected_from_source_graph_rels) = extract_nodes_and_relationships(source_entity_name, question, progress_callback)
            
            target_entity_label, target_entity = get_node_label(self.graph, target_entity_name)
            paths = query_direct(self.graph, target_entity, target_entity_label)
            if paths:
                (selected_from_target_paths,
                selected_from_target_nodes,
                selected_from_target_graph_rels) = select_paths2(paths, 
                                                        question, 
                                                        len(paths)//3, 
                                                        3, 
                                                        progress_callback)

            query_nodes = selected_from_target_nodes + selected_from_source_nodes

        else:            
            (source_to_target_paths, 
            from_target_paths, 
            from_source_paths) = result 

            if source_to_target_paths:
                (selected_from_target_paths, 
                selected_from_target_nodes, 
                selected_from_target_graph_rels) = select_paths2(from_target_paths, 
                                                                question, 
                                                                len(from_target_paths)//15, 
                                                                10, 
                                                                progress_callback)
            else:
                return None
        
            print("final_target_paths")
            print(len(selected_from_target_paths))

            if from_source_paths:
                (selected_from_source_paths, 
                selected_from_source_nodes, 
                selected_from_source_graph_rels) = select_paths2(from_source_paths, 
                                                                question, 
                                                                len(from_source_paths)//15,
                                                                10, 
                                                                progress_callback)
            else:
                selected_from_source_paths = []

            print("final_source_paths")
            print(len(selected_from_source_paths))

            if from_target_paths:
                (selected_source_to_target_paths, 
                selected_source_to_target_nodes, 
                selected_source_to_target_graph_rels) = select_paths2(source_to_target_paths, 
                                                                    question, 
                                                                    len(source_to_target_paths), 
                                                                    10, 
                                                                    progress_callback)
            else:
                selected_from_target_paths = []

            print("final_inter_relationships")
            print(len(selected_source_to_target_paths))
            query_nodes = selected_from_target_nodes + selected_from_source_nodes + selected_source_to_target_nodes

        if self.additional_entity_types is not None:
            additional_nodes = set()
            additional_paths = set()
            additional_graph_rels = set()

            for entityName, entityInfo in self.additional_entity_types.items():
                paths = query_direct(self.graph, entityName)
                if paths:
                    selected_paths, selected_nodes, selected_graph_rels = select_paths2(paths, 
                                                                                        question, 
                                                                                        len(paths)//15,
                                                                                        10, 
                                                                                        progress_callback)
                    
                    additional_paths.update(selected_paths)
                    additional_graph_rels.update(selected_graph_rels)
                    additional_nodes.update(selected_nodes)
                print("additional_entity_direct_graph_relationships")
                print(additional_paths)
            
            query_nodes += additional_nodes
        
        if self.additional_entity_types is not None:
            query_nodes += list(additional_nodes)

        query_nodes = (set(query_nodes))
        names_set = set(names_list)

        query_nodes = [name for name in query_nodes if name.lower() not in names_set]
        print("query nodes")
        print(len(query_nodes))
        print(query_nodes)

        mid_direct_paths = set()
        mid_direct_nodes = set()
        mid_direct_graph_rels = set()
        
        node_labels = get_node_labels_dict(self.graph, query_nodes)
        for node in query_nodes:
            node_label = node_labels.get(node)
            if node_label is not None:
                paths = query_direct(self.graph, node, node_label)
                if paths:
                    (selected_paths, 
                    selected_nodes, 
                    selected_graph_rels) = select_paths2(paths, 
                                                        question, 
                                                        len(paths)//3, 
                                                        3, 
                                                        progress_callback)
                    
                    mid_direct_paths.update(selected_paths)
                    mid_direct_nodes.update(selected_nodes)
                    mid_direct_graph_rels.update(selected_graph_rels)
                    print("success")
                    print(len(selected_paths))
                    print(selected_paths)
                    del paths, selected_paths, selected_nodes, selected_graph_rels
                    gc.collect()
                else:
                    print("skipping")
                    continue

        print("number of unique inter_direct_relationships:")
        print(len(mid_direct_paths))

        mid_inter_paths = query_between_direct(self.graph, 
                                                list(mid_direct_nodes), 
                                                query_nodes)
        
        (selected_mid_inter_paths, 
         selected_mid_inter_nodes, 
         selected_mid_inter_graph_rels) = select_paths2(mid_inter_paths, 
                                                        question, 
                                                        len(mid_inter_paths)//15, 
                                                        50, 
                                                        progress_callback)
        

        print("final_inter_direct_inter_relationships")
        print(len(selected_mid_inter_paths))

        all_graph_rels = set()
        all_graph_rels.update(selected_mid_inter_graph_rels)
        all_graph_rels.update(mid_direct_graph_rels)
        if selected_source_to_target_graph_rels is not None:
            all_graph_rels.update(selected_source_to_target_graph_rels)
        all_graph_rels.update(selected_from_source_graph_rels) 
        all_graph_rels.update(selected_from_target_graph_rels)

        if self.additional_entity_types is not None:
            all_graph_rels.update(additional_graph_rels)

        all_graph_rels = list(all_graph_rels)
        print("all_graph_rels")
        print(len(all_graph_rels))

########################################################################################################
        
        params = {
            "llm": self.llm, 
            "question": question,
            "source_list": list(selected_from_source_paths),
            "target_list": list(selected_from_target_paths),
            "inter_direct_list": list(mid_direct_paths),
            "inter_direct_inter": list(selected_mid_inter_paths),
            "source": names_list[0],
            "target": names_list[1]
        }

        if self.additional_entity_types is not None:
            params["additional_rels"] = additional_graph_rels
            print(f'additional_rels: {additional_graph_rels}')

        if selected_source_to_target_paths is not None:
            params["relationships_list"] = list(selected_source_to_target_paths)
            print(f'relationships_list: {selected_source_to_target_paths}')

        final_context = generate_answer(**params)

        answer = final_context

        response = {"result": answer, 
                    "all_rels": all_graph_rels}
                
        return response


