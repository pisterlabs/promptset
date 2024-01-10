import os
import openai
from utils.utils import get_project_root
from NER.spacy_ner import SpacyNER
from utils.logger import get_log_file, write_to_log  # Import the logger functions
from neo4j_api.neo4j_api import Neo4j_API
from openai_api.chat_test import single_chat as gpt_response
from openai_api.openai_client import OpenAI_API

def ner(input):
    """
    Where we would do NER on the next input.
    """
    print("Named Entity Recognition module:")

    ner = SpacyNER()
    disease_ner_results, scientific_entity_ner_results, pos_results, mesh_ids = ner.get_entities(input)

    # Look for mesh ids
    if mesh_ids:
        print("MESH IDS: {}".format(mesh_ids))
        disease_entities = [d.text for d in mesh_ids.keys()]

        # Get the mesh ids
        mesh_list = [mesh_ids[entity] for entity in mesh_ids.keys()]

        # Identify non-disease entities
        non_disease_entities = [entity for entity, e_type in scientific_entity_ner_results if
                                entity not in disease_entities]
        for entity, e_type in pos_results:
            if e_type == 'NOUN':
                in_diseases = False
                for d in disease_entities:
                    if entity in d:
                        in_diseases = True
                if not in_diseases:
                    non_disease_entities += [entity]

        relationships = []
        for entity, e_type in pos_results:

            if e_type == 'VERB':
                relationships += [entity]

    print("Non disease entities: {}".format(non_disease_entities))
    print("Relationships: {}".format(relationships))

    return mesh_ids, non_disease_entities, relationships

def get_gpt_response(single_entry, all_types, N=5):
    """
    To avoid the problem of getting too many different responses from the LLM, 
    try and aggregate them and take the most common response
    """
    responses = list()
    for i in range(N):
        # Specify prompt
        prompt = """Which of the following is {} most similar to in the following list: {}? 
        You may select an item even if it does not seem that similar, 
        just be sure to pick one. Only list the terms seperated by commas with 
        no additional information or descriptions.""".format(single_entry, all_types)
        # Append response to list
        response = gpt_response(prompt)
        responses.append()
    # Return the most common response
    return max(set(prompt), key=prompt.count)

def kg(ner_results):
    """
    This function identifies relevant nodes in the knowledge graph
    """

    mesh_ids, non_disease_entities, relationships = ner_results

    mesh_id_results = list()
    non_disease_ent_results = list()
    relationship_results = list()

    # Connect to the Neo4j API
    neo4j_api = Neo4j_API()

    # Check the MeSH terms are in the graph if any
    for mesh_id in mesh_ids:
        print(mesh_ids[mesh_id])
        mesh_query = "MATCH (n:MeSH_Disease {{name: 'MeSH_Disease:{}'}}) RETURN n LIMIT 25".format(mesh_ids[mesh_id][0])
        result = neo4j_api.search(mesh_query)
        mesh_id_results.append([mesh_ids[mesh_id][0], result])
    
    # Check the non-disease entities are in the graph if any
    node_types = neo4j_api.get_node_types()
    # Also be sure to save 
    for entity in non_disease_entities:
        non_disease_ent_results.append([entity, get_gpt_response(entity, node_types)])

    # Check the relationships are in the graph if any
    relationship_types = neo4j_api.get_rel_types()
    for rel in relationships:
        relationship_results.append([rel, get_gpt_response(rel, relationship_types)])
    
    return non_disease_ent_results, relationship_results

def start_chat(log_file=None):
    while True:
        # Get user input
        # user_input = input("User: ")
        user_input = "What drugs treat lung cancer?"

        # Identify entities
        ner_results = ner(user_input)

        # Identifies relevant nodes in the knowledge graph
        kg_results = kg(ner_results)

        # Send to Open AI API
        # response = call_openai_api(user_input)

        if log_file:
            write_to_log(log_file, "User: " + user_input)
            write_to_log(log_file, response)

"""
For the interface, use the Open AI class and create an object
From this object, use the LLM to make a query for neo4j
Test if the query returns anything
Keep going until the query returns something 
"""

class Prompt_Response():
    def __init__(self, user_input):
        self.user_input = user_input
        # NOTE: For testing, I added a single Mesh disease which I knew was in the KG
        self.mesh_ids = ['D044584']
        self.non_disease_entities = list()
        self.relationships = list()
        self.client = OpenAI_API()

    def perform_ner(self, debug=True):
        if debug:
            print("PERFORMING NER...")
        ner = SpacyNER()
        disease_ner_results, scientific_entity_ner_results, pos_results, mesh_ids = ner.get_entities(self.user_input)
        
        # Check for mesh_ids
        if mesh_ids:
            if debug:
                print("MESH IDS: {}".format(mesh_ids))
            disease_entities = [d.text for d in mesh_ids.keys()]

            # Get the mesh ids
            mesh_list = [mesh_ids[entity] for entity in mesh_ids.keys()]

            # Identify non-disease entities
            non_disease_entities = [entity for entity, e_type in scientific_entity_ner_results if
                                entity not in disease_entities]

            for entity, e_type in pos_results:
                if e_type == 'NOUN':
                    in_diseases = False
                    for d in disease_entities:
                        if entity in d:
                            in_diseases = True
                    if not in_diseases:
                        non_disease_entities += [entity]
            relationships = []
            for entity, e_type in pos_results:
                if e_type == 'VERB':
                    relationships += [entity]
        
        # TODO: Clean up double for loop
        for id in mesh_list:
            for i in id:
                self.mesh_ids.append(i)
        self.non_disease_entities += non_disease_entities
        self.relationships += relationships
        if debug:
            print("Debug diagnostic")
            print("self.mesh_ids: {}".format(self.mesh_ids))
            print("self.non_disease_entities: {}".format(self.non_disease_entities))
            print("self.relationships: {}".format(self.relationships))
    
    def process_kg_results(self, results: list):
        for res in results:
            print(res[1][0], type(res[1]))

    def kg(self, debug=True):
        # Save results for mesh, non-disease entities and relationships 
        # TODO: There is likely a better way of doing this
        mesh_id_results = list()
        non_empty_mesh_id_results = list()
        non_disease_ent_results = list()
        relationship_results = list()

        # Connect to the API for Neo4j
        neo4j_api = Neo4j_API()

        # Use the Neo4j api to find relevant mesh diseases
        for mesh_id in self.mesh_ids:
            # if debug:
            #     print(self.mesh_ids)
            # TODO: May need to use the API and check if there are better ways of calling this query
            # TODO: There may be Mesh items that are compounds instead of diseases
            mesh_query = "MATCH (n:MeSH_Disease {{name: 'MeSH_Disease:{}'}}) RETURN n LIMIT 25".format(mesh_id)
            result = neo4j_api.search(mesh_query)
            mesh_id_results.append([mesh_id, result])
        
        # Determine which Mesh IDs were able to be found in the knowledge graph
        # TODO: There is likely more that can be done with this information
        for id in mesh_id_results:
            if id[1][0] != []:
                non_empty_mesh_id_results.append(id[0])
        
        # TODO: Implement context queries to LLM
        # For each of the non-disease entities, see if you can create a query for it using the API
        # From there, try and see if the query returns anything
        # If the query returns something, then stop
        # Otherwise, try and update the prompt
            # There is no current way of saving the context
            # You must send the information every time in the prompt

        # Check the non-disease entities are in the graph if any
        node_types = list(neo4j_api.get_node_types())

        # Also be sure to save 
        for entity in self.non_disease_entities:
            found_word = False
            self.client.clear_context()
            while not found_word: 
                # If no message has been passed in yet, then start with this inital prompt 
                if self.client.get_context_length() < 2:
                    prompt = "What is the word '{}' closest to in the following list of terms: {}? You must select only one word from the list.".format(entity, node_types)
                # If there have been prompts before, then 
                else:
                    prompt = "Retry but do not make the output verbose at all."
                
                response = self.client.single_chat(prompt)
                
                if response[0][11:] in list(node_types):
                    # TODO: This may be redundant but add break just in case
                    found_word = True
                    break
                else:
                    # Add the context to the client for "failed" responses
                    context_message = response[0][11:]
                    self.client.add_context(context_message=context_message)
                print("LLM response: {}".format(response[0]))
                print("Parameter self.messages: {}".format(response[1]))
            # non_disease_ent_results.append([entity, get_gpt_response(entity, node_types)])
        

        

        
        

        
        

        
        if debug:
            print("MESH ID RESULTS")
            print(mesh_id_results)
            # prompt = "What does the MESH ID {} correspond to?".format(self.mesh_ids[0])
            # print(self.client.single_chat(prompt))
            # print(self.client.single_chat("What does this MESH ID correspond to: {}".format(self.mesh_ids[0])))

    # """
    # This function identifies relevant nodes in the knowledge graph
    # """

    # mesh_ids, non_disease_entities, relationships = ner_results
    # # Connect to the Neo4j API
    # neo4j_api = Neo4j_API()

    # # Check the MeSH terms are in the graph if any
    # for mesh_id in mesh_ids:
    #     print(mesh_ids[mesh_id])
    #     mesh_query = "MATCH (n:MeSH_Disease {{name: 'MeSH_Disease:{}'}}) RETURN n LIMIT 25".format(mesh_ids[mesh_id][0])
    #     result = neo4j_api.search(mesh_query)
    #     mesh_id_results.append([mesh_ids[mesh_id][0], result])
    
    # # Check the non-disease entities are in the graph if any
    # node_types = neo4j_api.get_node_types()
    # # Also be sure to save 
    # for entity in non_disease_entities:
    #     non_disease_ent_results.append([entity, get_gpt_response(entity, node_types)])

    # # Check the relationships are in the graph if any
    # relationship_types = neo4j_api.get_rel_types()
    # for rel in relationships:
    #     relationship_results.append([rel, get_gpt_response(rel, relationship_types)])
    

if __name__ == "__main__":
    prompt = "How does aspartate aminotransferase affect myocardial ischemia, arrhythmias, and ductal carcinoma?"
    pr = Prompt_Response(prompt)
    pr.perform_ner()
    pr.kg()


"""
Spare notes
        '''
        TODO Joseph
        - Replace 'Entity' with the Node Type identified from Neo4j API
            e.g., neo4j_api.get_node_type_properties() -> find the closest match. Maybe ask LLM to identify best match?
        - After creating the query, query the KG see if the node exists (if it's a class-based node like 'drug',
          then get a few examples? Otherwise if it's a specific drug with an ID, check it exists.

        MY NOTES
        Do ner on the results (which is what is currently doing) and see the results that are most most similar to 
        '''

        '''
        TODO Joseph
        - Similar to above, but use neo4j_api.get_rel_types() to find the closest match.
        - To consider, how do we know which node types the relationships needs? This means we have to look at the 
          original query, in the NER step and identify head and tail of the relationship... Then we can use the
          neo4j_api.get_uniq_relation_pairs() to find the closest match. 
        '''
"""