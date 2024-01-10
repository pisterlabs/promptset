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
    query_direct_constituents, 
    query_between_direct,
    get_node_labels_dict,
    get_node_label
)
from CustomLibrary.Graph_Utils import (
    select_paths2
)
from langchain.prompts import PromptTemplate
import gc
from CustomLibrary.OPC_Utils import pubchem_query, similar_pubchem_query
from CustomLibrary.Custom_Prompts import Graph_Answer_Gen_Template_alpaca


def generate_answer(llm, entities_list, question, start_paths, mid_paths, inter_direct_inter):
    prompt = PromptTemplate(template=Graph_Answer_Gen_Template_alpaca, input_variables=["input", "question"])
    #prompt = PromptTemplate(template=Graph_Answer_Gen_Template_alpaca, input_variables=["input", "question"])
    gen_chain = LLMChain(llm=llm, prompt=prompt)
    start_paths = ','.join(start_paths)
    Inter_relationships = mid_paths + inter_direct_inter
    Inter_sentences = ','.join(Inter_relationships)
    sep1 = f"Starting paths from {entities_list}:"
    sep2 = f"Intermediate paths of {entities_list}:"
    sentences = '\n'.join([sep1, start_paths, sep2, Inter_sentences])
    answer = gen_chain.run(input=sentences, question=question)
    print(answer)
    return answer

class OPC_GraphQA:
    def __init__(self, uri, username, password, llm, entities_list, constituents_dict, constituents_paths):
        self.graph = Graph(uri, auth=(username, password))
        self.llm = llm
        self.entities_list = entities_list
        self.constituents_dict = constituents_dict
        self.constituents_paths = constituents_paths

    def _call(self, question, progress_callback=None):
        start_paths = []
        start_nodes = []
        start_graph_rels = []

        for entity in self.entities_list:
            entity_label, entity_name = get_node_label(self.graph, entity)
            paths = query_direct(self.graph, entity_name, entity_label)
            if paths:
                (CKG_paths,
                CKG_nodes, 
                CKG_rels) = select_paths2(paths, 
                                            question, 
                                            len(paths)//3, 
                                            3, 
                                            progress_callback)
                start_paths.extend(CKG_paths)
                start_nodes.extend(CKG_nodes)
                start_graph_rels.extend(CKG_rels)

            if entity in self.constituents_dict and self.constituents_dict[entity]:
                constituents = self.constituents_dict[entity]
                constituents = [constituent for constituent in constituents if constituent != 'None']
                if constituents and 'None' not in constituents:  

                    for constituent in constituents:
                        constituent_label, constituent_name = get_node_label(self.graph, constituent)
                        paths = query_direct_constituents(self.graph, constituent_name, constituent_label)
                        
                        if paths:
                            (Constituent_CKG_paths,
                            Constituent_CKG_nodes, 
                            Constituent_CKG_rels) = select_constiuent_paths(entity,
                                                            paths, 
                                                            question, 
                                                            len(paths)//3, 
                                                            3, 
                                                            progress_callback)

                            start_paths.extend(Constituent_CKG_paths)
                            start_nodes.extend(Constituent_CKG_nodes)
                            start_graph_rels.extend(Constituent_CKG_rels)

                        pubchem_result = pubchem_query(entity, 
                                                constituent, 
                                                question, 
                                                progress_callback)
                        
                        if pubchem_result:
                            (pubchem_paths, 
                            pubchem_nodes, 
                            pubchem_rels) = pubchem_result

                            start_paths.extend(pubchem_paths)
                            start_nodes.extend(pubchem_nodes)
                            start_graph_rels.extend(pubchem_rels)
                        
                        similar_pubchem_result = similar_pubchem_query(entity,
                                                                        constituent,
                                                                        question,
                                                                        progress_callback)

                        if similar_pubchem_result:
                            (similar_pubchem_paths, 
                            similar_pubchem_nodes, 
                            similar_pubchem_rels) = similar_pubchem_result

                            start_paths.extend(similar_pubchem_paths)
                            start_nodes.extend(similar_pubchem_nodes)
                            start_graph_rels.extend(similar_pubchem_rels)
                else:   
                    continue
            else:
                continue

        if len(start_paths) == 0:
            return None

        print("start_paths:", len(start_paths))
        print("start_nodes:", len(start_nodes))
        print("start_graph_rels:", len(start_graph_rels))

        query_nodes = list(set(start_nodes))

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
        

        n_cluster = max(len(mid_inter_paths)//10, 1)
        if n_cluster > 30:
            n_embed = 30
        else:
            n_embed = n_cluster

        print("n_cluster:", n_cluster)
        (selected_mid_inter_paths, 
         selected_mid_inter_nodes, 
         selected_mid_inter_graph_rels) = select_paths2(mid_inter_paths, 
                                                        question, 
                                                        n_cluster, 
                                                        n_embed,
                                                        progress_callback)
        
        print("final_inter_direct_inter_relationships")
        print(len(selected_mid_inter_paths))

        all_graph_rels = set()
        all_graph_rels.update(selected_mid_inter_graph_rels)
        all_graph_rels.update(mid_direct_graph_rels)
        all_graph_rels.update(start_graph_rels)

        all_graph_rels = list(all_graph_rels)
        print("all_graph_rels")
        print(len(all_graph_rels))

########################################################################################################
        
        params = {
            "llm": self.llm,
            "entities_list": self.entities_list,
            "question": question,
            "start_paths": start_paths,
            "mid_paths": list(mid_direct_paths),
            "inter_direct_inter": list(selected_mid_inter_paths),
        }

        final_context = generate_answer(**params)

        answer = final_context

        response = {"result": answer,
                    "all_rels": all_graph_rels}
                
        return response


