from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain.chains.llm import LLMChain
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from py2neo import Graph
import numpy as np
from langchain.prompts import PromptTemplate
import gc

from CustomLibrary.Graph_Queries import (
    query_direct, 
    query_between_direct,
    get_node_labels_dict,
    )
from CustomLibrary.Graph_Utils import (
    select_paths, 
    select_paths2, 
)
from CustomLibrary.Custom_Prompts import Graph_Answer_Gen_Template, Graph_Answer_Gen_Template2_alpaca

from CustomLibrary.OpenTargets import (
    query_disease_info,
    query_drug_info,
    query_target_info,
    query_predicted_drug_info
)

def generate_answer(llm, source_list, target_list, inter_direct_list, inter_direct_inter, question, previous_answer, source, target, additional_list:Optional[List[str]]=None):
    prompt = PromptTemplate(template=Graph_Answer_Gen_Template2_alpaca, input_variables=["input", "question", "previous_answer"])
    #prompt = PromptTemplate(template=Graph_Answer_Gen_Template_alpaca, input_variables=["input", "question"])
    gen_chain = LLMChain(llm=llm, prompt=prompt)
    source_rels = ', '.join(source_list)
    target_rels = ','.join(target_list)
    multi_hop_rels = inter_direct_list + inter_direct_inter
    multi_hop_sentences = ','.join(multi_hop_rels)
    sep_1 = f"Direct paths from {source}:"
    sep2 = f"Direct paths from {target}:"
    sep3 = f"Paths between the nodes in the direct paths of {source} and {target}:"
    if additional_list:
        additional_sentences = ','.join(additional_list)
        sep4 = f"Additional relations related to the question"
        sentences = '\n'.join([sep_1, source_rels, sep2, target_rels, sep3, multi_hop_sentences, sep4, additional_sentences])
    else:
        sentences = '\n'.join([sep_1, source_rels, sep2, target_rels, sep3, multi_hop_sentences])
    answer = gen_chain.run(input=sentences, question=question, previous_answer=previous_answer)
    print(answer)
    return answer

class OpenTargetsGraphQA:
    def __init__(self, uri, username, password, llm, entity_types, additional_entity_types=None):
        self.graph = Graph(uri, auth=(username, password))
        self.llm = llm
        self.entity_types = entity_types
        self.additional_entity_types = additional_entity_types  # Store the additional entity types dictionary

    def _call(self, names_list, question, previous_answer, progress_callback=None):
        
        entities_list = list(self.entity_types.items())

        # The first entity is the source entity
        source_entity_name, source_entity_type = entities_list[0]

        # The second entity is the target entity
        target_entity_name, target_entity_type = entities_list[1]

        if source_entity_type == "Disease":
            from_source_paths, source_nodes, from_source_graph_rels = query_disease_info(source_entity_name, question)
        if source_entity_type == "Drug":
            from_source_paths, source_nodes, from_source_graph_rels= query_drug_info(source_entity_name, question)
        if source_entity_type == "Gene":
            from_source_paths, source_nodes, from_source_graph_rels = query_target_info(source_entity_name, question)

        if target_entity_type == "Disease":
            from_target_paths, target_nodes, from_target_graph_rels = query_disease_info(target_entity_name, question)
        if target_entity_type == "Drug":
            from_target_paths, target_nodes, from_target_graph_rels= query_drug_info(target_entity_name, question)
        if target_entity_type == "Gene":
            from_target_paths, target_nodes, from_target_graph_rels = query_target_info(target_entity_name, question)

        query_nodes = source_nodes + target_nodes
        query_nodes = set(list(query_nodes))
        graph_rels = set(graph_rels)

        
        if self.additional_entity_types is not None:
            additional_paths = set()
            additional_graph_rels = set()
            for entityname, entity_info in self.additional_entity_types.items():
                if entity_info is not None and entity_info['entity_type'] is not None:  
                    entity_type = entity_info['entity_type'][1]  # Extract entity type from the nested dictionary
                    if entity_type == "Disease":
                        paths, nodes, graph_rels = query_disease_info(entityname, question)
                    elif entity_type == "Drug":
                        paths, nodes, graph_rels = query_predicted_drug_info(entityname, question)
                    elif entity_type == "Gene":
                        paths, nodes, graph_rels = query_target_info(entityname, question)
                    additional_paths.update(paths)
                    query_nodes.update(nodes)
                    additional_graph_rels.update(graph_rels)

        print("query nodes")
        print(len(query_nodes))
        print(query_nodes)

        mid_direct_paths = set()
        mid_direct_nodes = set()
        mid_direct_graph_rels = set()
        query_nodes = list(query_nodes)
        node_labels = get_node_labels_dict(self.graph, query_nodes)
        for node in query_nodes:
            paths = []  # Initialize paths to an empty list
            node_label = node_labels.get(node)
            if node_label is not None:
                paths = query_direct(self.graph, node, node_label)
            if paths:
                (selected_paths, 
                selected_nodes, 
                selected_graph_rels) = select_paths2(paths, 
                                                    question, 
                                                    max(1, len(paths)//3), 
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
        all_graph_rels.update(from_source_graph_rels)
        all_graph_rels.update(from_target_graph_rels)
        
        if self.additional_entity_types is not None:
            all_graph_rels.update(additional_graph_rels)

        all_graph_rels = list(all_graph_rels)

        print("all_graph_rels")
        print(len(all_graph_rels))

########################################################################################################
             
        params = {
            "llm": self.llm, 
            "question": question,
            "source_list": list(from_source_paths),
            "target_list": list(from_target_paths),
            "inter_direct_list": list(mid_direct_paths),
            "inter_direct_inter": list(selected_mid_inter_paths),
            "source": names_list[0],
            "target": names_list[1],
            "previous_answer": previous_answer
        }

        if self.additional_entity_types is not None:
            params["additional_rels"] = additional_paths

        final_context = generate_answer(**params)

        answer = final_context

        response = {"result": answer, 
                    "all_rels": all_graph_rels}
                
        return response