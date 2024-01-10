import requests
from qwikidata.entity import WikidataItem
from qwikidata.linked_data_interface import get_entity_dict_from_api
import json
from tqdm import tqdm
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
import re
import pinecone

class KRAFTWikidataAPI:

    """
    Read the pid_to_label.json file and create a mapping from label to pid.
    Creating this mapping speeds up the process of choosing properties.
    """

    with open("src/pid_to_label.json", "r") as f:
        pid_to_info = json.load(f)

    label_to_pid = {}
    for pid, (label, _, _) in tqdm(pid_to_info.items()):
        label_to_pid[label] = pid

    @staticmethod
    def extract_entity_labels(llm, question):
        """
        Extracts entity labels from a given question using a language model.

        Parameters:
        - llm: A language model instance.
        - question: The question text from which to extract entities.

        Returns:
        - A list of entity labels extracted from the question.
        """
        
        # Set up the LLMChain prompts for extracting entities
        extract_template = "To answer this question: \"{query}\", what few Wikipedia pages could be used? Don't answer the question, just list the entities. Give ONLY a comma-separated list of their labels."
        extract_prompt = PromptTemplate(template=extract_template, input_variables=["query"])
        extract_llm_chain = LLMChain(prompt=extract_prompt, llm=llm)

        # Extract entities from the question via prompting
        response = extract_llm_chain.run(question)
        print("EXTRACT RESPONSE:" + response)

        # Interpret the response as a comma-separated list of entity labels
        entity_labels = [label.strip() for label in response.split(",")]
        return entity_labels


    @staticmethod
    def get_wikidata_entity_id(search_term):
        """
        Retrieves the Wikidata entity ID for a given search term.

        Parameters:
        - search_term: The term to search in Wikidata.

        Returns:
        - The Wikidata entity ID if found, otherwise None.
        """
        # Use the Wikidata SPARQL API to search for the entity ID
        url = 'https://query.wikidata.org/sparql'
        query = '''
        SELECT ?item WHERE {
          ?item rdfs:label "%s"@en.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }
        ''' % search_term
        try:
            response = requests.get(url, params={'format': 'json', 'query': query})
            data = response.json()
            items = data['results']['bindings']
            if items:
                return items[0]['item']['value'].split('/')[-1]
            else:
                return None
        except:
            print(f"Failed {search_term}")
            return None
        
    @staticmethod
    def get_entity_dict(entity_id):
        """
        Fetches the entity dictionary from Wikidata for a given entity ID.

        Parameters:
        - entity_id: The Wikidata entity ID.

        Returns:
        - A dictionary representation of the entity.
        """
        return get_entity_dict_from_api(entity_id)

    @staticmethod
    def get_snaks_with_labels(entity_dict):
        """
        Extracts snaks (statements/claims) with labels from a Wikidata entity.

        Parameters:
        - entity_dict: The dictionary representation of a Wikidata entity.

        Returns:
        - A dictionary of snaks with their labels and descriptions.
        """
        # Initialize the WikidataItem
        wikidata_item = WikidataItem(entity_dict)

        # Extract the snaks with labels and descriptions
        snaks = {}
        for property_id, claims in wikidata_item.get_truthy_claim_groups().items():
            if property_id not in KRAFTWikidataAPI.pid_to_info:
                # Unlikely to happen, but just in case, since LLMs are not perfect
                continue

            snaks[property_id] = {}
            snaks[property_id]["label"] = KRAFTWikidataAPI.pid_to_info[property_id][0]
            snaks[property_id]["description"] = KRAFTWikidataAPI.pid_to_info[property_id][1]
            snaks[property_id]["entities"] = []

            # For each property, there are many claims
            for claim in claims:
                # A snak is the (property, value) pair of a claim
                if claim.mainsnak.datavalue is not None:
                    snaks[property_id]["entities"].append(claim.mainsnak.datavalue.value)

        return snaks
    
    @staticmethod
    def get_entity_ids(entity_labels):
        """
        Converts a list of entity labels to their corresponding Wikidata IDs.

        Parameters:
        - entity_labels: A list of entity labels.

        Returns:
        - A list of corresponding Wikidata entity IDs.
        """
        entity_ids = []
        for entity_label in entity_labels:
            entity_id = KRAFTWikidataAPI.get_wikidata_entity_id(entity_label)
            # If the entity ID is not found, skip it. Ideally, this should not happen, but Exact Match queries often fail.
            if entity_id is not None:
                entity_ids.append(entity_id)
        return entity_ids
    
    @staticmethod
    def choose_properties(llm, embedding_model, question, entity_dict, snaks, choose_type='classic', choose_count=3):
        """
        Chooses relevant Wikidata properties for a given question and entity.

        Parameters:
        - llm: A language model instance.
        - embedding_model: A model for generating embeddings.
        - question: The question for which properties are to be chosen.
        - entity_dict: The dictionary representation of the entity.
        - snaks: The snaks associated with the entity.
        - choose_type: The method for choosing properties ('classic' or 'nearest_neighbor').
        - choose_count: The number of properties to choose.

        Returns:
        - A list of chosen property IDs.
        """
        entity_label = entity_dict['labels']['en']['value']
        print()
        print(entity_label)

        # This choose_type uses an LLM to pick the relevant queries
        if choose_type == 'classic':

            # Set up the LLMChain prompts for choosing properties
            template = "To answer this question: \"{query}\", which of these are the top \"{k}\" relevant Wikidata properties for the subject \"{subject}\" out of the comma-separated properties \"{properties}\"?\n\n Give a comma-separated list, without quotations, of the top \"{k}\" labels you think are most relevant for answering the question."
            prompt = PromptTemplate(template=template, input_variables=["query", "k", "subject", "properties"])
            llm_chain_pipeline_classic = LLMChain(prompt=prompt, llm=llm)
            
            # Prepare the properties for the prompt to choose from
            properties = ', '.join([snak["label"] for snak in snaks.values() if snak["label"] is not None])
            response = llm_chain_pipeline_classic.run(query=question, k=choose_count, subject="subject", properties=properties)


            # Interpret the response as a comma-separated list of property labels
            edge_pids = []
            for edge_label in response.split(","):
                stripped_edge_label = edge_label.strip()
                if stripped_edge_label in KRAFTWikidataAPI.label_to_pid:
                    edge_pids.append(KRAFTWikidataAPI.label_to_pid[stripped_edge_label])
                else:
                    # Unlikely to happen, but just in case, since LLMs are not perfect
                    continue
    
        elif choose_type == 'nearest_neighbor':

            # Set up the LLMChain prompt to get the query whose embedding will be used for nearest neighbor search
            template = "To answer this question: \"{query}\", what is the 1 top query you have for subject \"{subject}\"? Output only the query"
            prompt = PromptTemplate(template=template, input_variables=["query", "subject"])
            llm_chain_pipeline_nn = LLMChain(prompt=prompt, llm=llm)

            # Get the query and its embedding
            query_pids = list(snaks.keys())
            query = llm_chain_pipeline_nn.run(query=question, subject=entity_label).strip()
            query_embedding = embedding_model.embed_documents([query])[0]

            # Use pinecone to find the nearest neighbors of the query embedding
            pinecone_model = pinecone.init(api_key="53ad2ca2-3d03-45a3-b4c3-36baa6ca835a", environment="gcp-starter")
            pinecone_index = pinecone.Index('kraft')
            query_result = pinecone_index.query(vector=query_embedding, top_k=choose_count, include_values=False, include_metadata=False, filter={"pid": {'$in': query_pids}})
            edge_pids = [match['id'] for match in query_result['matches']]
        
        else:
            raise ValueError("Invalid choose_type: " + choose_type)
        
        return edge_pids
    
kw = KRAFTWikidataAPI