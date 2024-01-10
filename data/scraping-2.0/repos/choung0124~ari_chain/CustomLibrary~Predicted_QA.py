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
    get_node_labels_dict
    )

from CustomLibrary.Graph_Utils import (
    select_paths, 
    select_paths2, 
)
from CustomLibrary.Custom_Prompts import (
    Graph_Answer_Gen_Template3, 
    Graph_Answer_Gen_Template3_alpaca
)

from CustomLibrary.OpenTargets import(
    query_predicted_disease_info,
    query_predicted_target_info,
    query_predicted_drug_info
)

def generate_answer(llm, source_list, target_list, inter_direct_list, inter_direct_inter, question, previous_answer, source, target, additional_list:Optional[List[str]]=None):
    prompt = PromptTemplate(template=Graph_Answer_Gen_Template3_alpaca, input_variables=["input", "question", "previous_answer"])
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
        sep4 = f"Direct paths from "
        sentences = '\n'.join([sep_1, source_rels, sep2, target_rels, sep3, multi_hop_sentences, sep4, additional_sentences])
    else:
        sentences = '\n'.join([sep_1, source_rels, sep2, target_rels, sep3, multi_hop_sentences])
    answer = gen_chain.run(input=sentences, question=question, previous_answer=previous_answer)
    print(answer)
    return answer

class PredictedGrqphQA:
    def __init__(self, uri, username, password, llm, entity_types, question, additional_entity_types=None):
        self.graph = Graph(uri, auth=(username, password))
        self.llm = llm
        self.entity_types = entity_types
        self.additional_entity_types = additional_entity_types  # Store the additional entity types dictionary
        entities_list = list(self.entity_types.items())

        # The first entity is the source entity
        source_entity_name, source_entity_type = entities_list[0]

        # The second entity is the target entity
        target_entity_name, target_entity_type = entities_list[1]

        self.source_result_dict = {}
        self.target_result_dict = {}

        if source_entity_type == "Disease":
            self.source_result_dict = query_predicted_disease_info(source_entity_name, question)
            self.source_similar_entity_names = list(self.source_result_dict.keys())

        if source_entity_type == "Drug":
            self.source_result_dict = query_predicted_drug_info(source_entity_name, question)
            self.source_similar_entity_names = list(self.source_result_dict.keys())

        if source_entity_type == "Gene":
            self.source_result_dict = query_predicted_target_info(source_entity_name, question)
            self.source_similar_entity_names = list(self.source_result_dict.keys())

        if target_entity_type == "Disease":
            self.target_result_dict = query_predicted_disease_info(target_entity_name, question)
            self.target_similar_entity_names = list(self.target_result_dict.keys())

        if target_entity_type == "Drug":
            self.target_result_dict = query_predicted_drug_info(target_entity_name, question)
            self.target_similar_entity_names = list(self.target_result_dict.keys())

        if target_entity_type == "Gene":
            self.target_result_dict = query_predicted_target_info(target_entity_name, question)
            self.target_similar_entity_names = list(self.target_result_dict.keys())
        
        # Initialize dictionaries for storing result dictionaries and key lists
        self.additional_result_dicts = {}
        self.additional_similar_entity_names_lists = {}
        self.additional_similar_entity_names = {}  # Add this line

        if self.additional_entity_types is not None:
            for entityname, entity_info in self.additional_entity_types.items():
                if entity_info is not None and entity_info['entity_type'] is not None:  
                    entity_type = entity_info['entity_type'][1]  # Extract entity type from the nested dictionary
                    if entity_type == "Disease":
                        self.additional_result_dict = query_predicted_disease_info(entityname, question)
                    elif entity_type == "Drug":
                        self.additional_result_dict = query_predicted_drug_info(entityname, question)
                    elif entity_type == "Gene":
                        self.additional_result_dict = query_predicted_target_info(entityname, question)
                    
                    # Store the result dictionary and list of keys for this additional entity
                    self.additional_result_dicts[entityname] = self.additional_result_dict
                    self.additional_similar_entity_names_lists[entityname] = list(self.additional_result_dict.keys())


    def _call(self, names_list, question, previous_answer, progress_callback=None):
        # Get the maximum length among all lists of keys
        if self.additional_entity_types is not None:
            max_length = max(len(self.source_similar_entity_names), len(self.target_similar_entity_names), max([len(names) for names in self.additional_similar_entity_names_lists.values()]))
        else:
            max_length = max(len(self.source_similar_entity_names), len(self.target_similar_entity_names))
            
        # Iterate over the range of the maximum length
        for i in range(max_length):
            source_nodes = source_rels = target_nodes = target_rels = []
            if i < len(self.source_similar_entity_names) and i < len(self.target_similar_entity_names):
                print(self.target_similar_entity_names)
                
                names_to_print = [self.source_similar_entity_names[i], self.target_similar_entity_names[i]]
                
                name = self.source_similar_entity_names[i]
                source_paths = self.source_result_dict[name]['paths']
                source_nodes = self.source_result_dict[name]['nodes']
                source_rels = self.source_result_dict[name]['rels']
                
                name = self.target_similar_entity_names[i]
                target_paths = self.target_result_dict[name]['paths']
                target_nodes = self.target_result_dict[name]['nodes']
                target_rels = self.target_result_dict[name]['rels']
            
            # Initialize query_nodes, graph_rels and final_additional_paths with source and target nodes and relationships
            query_nodes = source_nodes + target_nodes
            graph_rels = source_rels + target_rels
            final_additional_paths = []

            if self.additional_entity_types is not None:
                for entityname in self.additional_entity_types:
                    additional_similar_entity_names = self.additional_similar_entity_names_lists[entityname]

                    if i < len(additional_similar_entity_names):
                        name = additional_similar_entity_names[i]
                        
                        additional_result_dict = self.additional_result_dicts[entityname]
                        additional_paths = additional_result_dict[name]['paths']
                        additional_nodes = additional_result_dict[name]['nodes']
                        additional_rels = additional_result_dict[name]['rels']
                        query_nodes += additional_nodes
                        graph_rels += additional_rels
                        final_additional_paths += additional_paths
                        names_to_print.append(name)
                    else:
                        names_to_print.append(None)  # Append None or some default value when no more entities are available

            print("query nodes")
            print(len(query_nodes))
            print(query_nodes)
            if query_nodes is None or len(query_nodes) == 0:
                response = {"result": None, 
                            "all_rels": None,
                            "names_to_print": None}
                return response

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
            all_graph_rels.update(source_rels) 
            all_graph_rels.update(target_rels)
            
            if self.additional_entity_types is not None:
                all_graph_rels.update(additional_rels)

            all_graph_rels = list(all_graph_rels)

            print("all_graph_rels")
            print(len(all_graph_rels))


    ########################################################################################################
            
            params = {
                "llm": self.llm, 
                "question": question,
                "source_list": list(source_paths),
                "target_list": list(target_paths),
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
                        "all_rels": all_graph_rels,
                        "names_to_print": names_to_print}
                    
            yield response