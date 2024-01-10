import os
from decouple import config
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import json
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import spacy
import pickle


class VirtualAgent(object):

    def __init__(self,
                 modes = ['Depression']):

        self.modes = modes

    def create_vector_db(self,partitions = 3,chunk_size = 3):

        nlp = spacy.load('SpacyModels')
        model = SentenceTransformer('all-mpnet-base-v2',cache_folder='VectorizationModels')
        vector_db = dict()
        for mode in self.patient_data:

            print ('indexing patient records ... ')
            
            for patient_record in tqdm(self.patient_data[mode]):

                print ('indexing record', patient_record)
                
                patient_text = self.patient_data[mode][patient_record]
                doc = nlp(patient_text[0])
                sents = [sent.text for sent in doc.sents]
                n_sents = len(sents)
                chunked_sents = [' '.join(sents[i:i+2]) for i in range(n_sents-2)]
                
                for chunk in chunked_sents:
                    vector_db[patient_record,chunk] = (model.encode([chunk]))
        
        with open('VectorDatabase/patientDb.db', 'ab') as dbfile:
            pickle.dump(vector_db, dbfile)                    

    def build_knowledge_graphs(self):

        for mode in self.modes:

            patient_records = os.listdir("PatientAssessments/"+mode+"/")
            for patient_record in patient_records:

                if patient_record == 'KnowledgeGraphs':
                    continue

                with open("PatientAssessments/"+mode+"/"+patient_record) as f:
                    patient_record_dict = dict(json.load(f))
                    patient_triples = patient_record_dict['triples']

                    source = [triple[0] for triple in patient_triples]
                    target = [triple[-1] for triple in patient_triples]
                    relations = [triple[1] for triple in patient_triples]

                    kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})

                    patient_graph = nx.from_pandas_edgelist(kg_df, "source", "target",
                                                            edge_attr=True, create_using=nx.MultiDiGraph())
                    
                    plt.figure(figsize=(18,18))
                    pos = nx.spring_layout(patient_graph)
                    nx.draw(patient_graph, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos)

                    #save the graph
                    plt.savefig("PatientAssessments/"+mode+"/KnowledgeGraphs/"+patient_record.split('.')[0]+".png")


    def perform_patient_assessments(self):

        
        for mode in self.patient_data:

            print ('processing patient records ... ')

            for patient_record in tqdm(self.patient_data[mode]):

                print ('processing record', patient_record)
                
                patient_text = self.patient_data[mode][patient_record]

                os.environ["OPENAI_API_KEY"] = config('KEY')

                '''
                intent_template = (
                "Act as the intent classification component of a home assistant, similar to Amazon Alexa "
                "(except your name is 'Becca', not 'Alexa').\n"
                f"Common intents include: {', '.join(self.INTENTS)}, ...\n"
                'You receive input in json format: `{{"input":...}}`\n'
                'You respond in json format: `{{"intents": [ ... , ... , ] , "exact input substrings": [ ... , ... , ] }}`\n'
                'Multiple intents are allowed`\n\n'
                '{{"input":{spoken_request}}}'
                )

                llm = OpenAI(temperature=0.0)
                prompt = PromptTemplate(
                input_variables=["spoken_request"],
                template=intent_template,
                )

                intents_and_explanations = llm(prompt.format(spoken_request=patient_text))
                '''
                
                '''
                assessment_template = ("Act as a virtual health assessment assistant`\n"
                                       "You will summarize the text based on a set of intents clearly and in your own words`\n"
                                       f"Common intents include: {', '.join(self.INTENTS)}, ...\n"
                                       'You receive input in json format: `{{"input":...}}`\n'
                                       'You respond in json format: `{{"patient assessment summary": ... }}`\n'
                                       'Multiple intents are allowed`\n\n'
                                       '{{"input":{spoken_request}}}'
                                       )

                llm2 = OpenAI(temperature=0.0, max_tokens = 512)
                prompt = PromptTemplate(
                input_variables=["spoken_request"],
                template=assessment_template,
                )

                patient_assessment = llm2(prompt.format(spoken_request=patient_text))
                '''

                knowledge_graph_template = ("Act as a virtual health assessment assistant`\n"
                                       "You will summarize the text as knowledge graph triples based on a set of intents`\n"
                                       f"Common intents include: {', '.join(self.INTENTS)}, ...\n"
                                       'You receive input in json format: `{{"input":...}}`\n'
                                       'You respond in json format: `{{"triples": [ [intent1, "reason", exact_substring1], [intent2, "reason", exact_substring2], ... , ] }}`\n'
                                       'Multiple intents are allowed`\n\n'
                                       '{{"input":{spoken_request}}}'
                                       )
                
                llm3 = OpenAI(temperature=0.0, max_tokens = 512)
                prompt = PromptTemplate(
                input_variables=["spoken_request"],
                template=knowledge_graph_template,
                )

                triples = llm3(prompt.format(spoken_request=patient_text))
                
                try:
                    
                    '''
                    json_response_as_dict = dict(json.loads(intents_and_explanations.strip()))
                    json_response_as_dict.update(dict(json.loads(patient_assessment.strip())))
                    json_response_as_dict.update(dict(json.loads(triples.strip())))
                    '''

                    #json_response_as_dict = dict(json.loads(patient_assessment.strip()))
                    #json_response_as_dict.update(dict(json.loads(triples.strip())))
                    json_response_as_dict = dict(json.loads(triples.strip()))

                    with open("PatientAssessments/"+mode+"/"+patient_record.split('.')[0]+".json", "w") as outfile: 
                        json.dump(json_response_as_dict, outfile)
                    

                except Exception as e:
                    print ("an error occured")

    def load_patient_data(self):

        self.patient_data = dict()
        if 'Depression' in self.modes:
            self.patient_data['Depression'] = dict()

            patient_file_names = os.listdir('PatientRepository/Depression/')

            for file_name in patient_file_names:

                with open('PatientRepository/Depression/'+file_name) as f:
                    self.patient_data['Depression'][file_name] = [item.lower() for item in f.read().splitlines()]

    def get_patient_data(self):

        return self.patient_data

    def load_intents(self):

        self.INTENTS = list()
        if 'Depression' in self.modes:

            with open('KnowledgeRepository/Depression.txt') as f:
                self.INTENTS += [item.lower() for item in f.read().splitlines()]

    def get_intents(self):
        return self.INTENTS