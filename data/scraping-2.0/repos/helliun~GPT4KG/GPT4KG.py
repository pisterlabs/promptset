import json
import pydot
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import openai
from PIL import Image

system_text = """You are an expert AI that extracts knowledge graphs from text and outputs JSON files with the extracted knowledge, and nothing more. Here's how the JSON is broken down.
Entity dictionaries are organized in a list
Every entity mentioned in the text has its own entity dictionary, in which the name of the entity is the key, and the value is a list of relationships.
Each relationship contains a short word or two accurately describing the relationship to the other entity as the key, and then the other entity as a value.
All inverses of these relationships are represented in the relationship list of the other entities. This is REALY IMPORTANT. For example if Apple created the iPhone, it is also important to note that the iPhone was created by Apple (each entity should have this relsationship from their perspective).
Non specified relationships are also inferred (if person X is the son of person Y, and person Z is person X's sibling, person Z is also the child of person Y).
The JSON contains NO NEW LINES. All the data should be on one line.
Every entity has a "description" relationship which provides a short description of what it is in a few words. If the description references another entity, then this relationship MUST be graphed, even if it is redundant.
Relationships are only created about facts, not just any connection between two entities mentioned in the text.
Example output:
[{"Toki Pona": [{"description": "philosophical artistic constructed language"}, {"translated as": "the language of good"}, {"created by": "Sonja Lang"}, {"first published": "2001"}, {"complete form published in": "Toki Pona: The Language of Good"}, {"supplementary dictionary": "Toki Pona Dictionary"}], "Sonja Lang": [{"description": "Canadian linguist and translator"}, {"creator of": "Toki Pona"}], "Toki Pona: The Language of Good": [{"description": "book"}, {"published in": "2014"}, {"language": "Toki Pona"}], "Toki Pona Dictionary": [{"description": "dictionary"}, {"released in": "July 2021"}, {"based on": "community usage"}]}]"""

class KnowledgeGraph:
    def __init__(self,api_key,kg_file=""):
        openai.api_key = api_key 
        self.graph = pydot.Dot(graph_type="digraph")
        self.entities = {}
        self.fact_scores = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        self.entity_embeddings = {}
        if kg_file!="":
          self.load_graph(kg_file)

    def add_entity(self, name, description):
        if name not in self.entities:
            self.entities[name] = {"description": description}
            entity_node = pydot.Node(name, label=f"{name}\n({description})")
            self.graph.add_node(entity_node)
            self.entity_embeddings[name] = self.model.encode(name)#+": \n"+"\n".join([key+": "+kg.entities[name][key] for key in kg.entities[name]]))
            print("added embedding")
            
    def add_relationship(self, entity1, relationship, entity2):
        if entity1 in self.entities:
            try: 
              self.entities[entity1][relationship] += ", "+entity2
            except:
              self.entities[entity1][relationship] = entity2
            edge = pydot.Edge(entity1, entity2, label=relationship)
            self.graph.add_edge(edge)

    def update_graph(self, json_str,clean=True):
        try:
          data = json.loads(json_str)
        except:
          print("GPT4 failed to create a valid JSON. Input may be too long for processing.")
          return
        for entity_dict in data:
            for entity, relationships in entity_dict.items():
                try:
                  self.add_entity(entity, relationships[0]["description"])
                except:
                  self.add_entity(entity, "")
                for rel in relationships[1:]:
                    for relationship, other_entity in rel.items():
                        try:
                          self.add_relationship(entity, relationship, other_entity)
                        except:
                          for o in other_entity:
                            self.add_relationship(entity, relationship, o)
        if clean:
          for entity_dict in data:
              for entity, relationships in entity_dict.items():
                self.clean_graph(entity)

    def display_graph(self, output_file="knowledge_graph.png"):
        self.graph.write_png(output_file)
        img = Image.open(output_file)
        img.show()
        return img

    def search(self, query, n=5):
      if len(self.entity_embeddings)<5:
        n = len(self.entity_embeddings)
      query_embedding = self.model.encode(query)
      query_tensor = torch.tensor([query_embedding])
      entity_tensor = torch.tensor(list(self.entity_embeddings.values()))
      similarities = util.cos_sim(query_tensor, query_tensor).numpy()
      top_indices = np.argsort(similarities[0])[-n:][::-1]
      results = [(list(self.entity_embeddings.keys())[index], similarities[0][index]) for index in top_indices]
      return results

    def related_entities(self,query, n=5):
      query_embedding = self.model.encode(query)
      query_tensor = torch.tensor([query_embedding])
      potentities = [key+": "+self.entities[key]["description"] for key in self.entities]
      entity_tensor = self.model.encode(potentities)
      similarities = util.cos_sim(query_tensor, entity_tensor).numpy()
      if len(similarities)<n:
        n = len(similarities)
      top_indices = np.argsort(similarities[0])[-n:][::-1]
      results = [potentities[index] for index in top_indices]
      return results

    def text_to_data(self,text):
      system = {"role":"system","content":system_text}
      messages = [system]
      try:
        related = self.related_entities(text)
        text = text+f"\n\nGenerate the JSON for the text above, remembering to add inverse relationships and inferences. Here are some related entities already in the graph. If you are adding information about any of them, refer to them by the names below (otherwise ignore this information):\n\n{str(related)}"
      except:
        pass
      messages.append({"role":"user","content":text})
      output = openai.ChatCompletion.create(model="gpt-4",messages=messages)["choices"][0]["message"].to_dict()["content"]
      return output

    def learn(self,text,show_output=False):
      json_str = self.text_to_data(text)
      if show_output:
        print(json_str)
      self.update_graph(json_str)

    def graph_search(self,query,n=5,path="subgraph.png"):
      results = self.search(query, n)
      if len(results)<n:
        n = len(results)
      top_ents = [results[i][0] for i in range(n)]
      data = [{ent:[{key:self.entities[ent][key]} for key in self.entities[ent]]} for ent in top_ents]
      new = KnowledgeGraph()
      json_string = json.dumps(data) 
      new.update_graph(str(json_string),clean=False) 
      new.display_graph(path)
    
    def text_search(self,query,n=3):
      results = self.search(query, n)
      keys = [r[0] for r in results]
      potentities = [key+": "+str(self.entities[key]) for key in keys]
      for p in potentities:
        print(p)
    
    def qa_search(self,query,n=5):
      results = self.search(query, n)
      keys = [r[0] for r in results]
      facts = [key+": "+str(rel).replace("description","is")+" "+str(self.entities[key][rel]) for key in keys for rel in self.entities[key]]
      query_embedding = self.model.encode(query)
      query_tensor = torch.tensor([query_embedding])
      fact_tensor = self.model.encode(facts)
      similarities = util.cos_sim(query_tensor, fact_tensor).numpy()
      if len(similarities[0])<n:
        n = len(similarities)
      top_indices = np.argsort(similarities[0])[-n:][::-1]
      results = [facts[index] for index in top_indices]
      return results
    
    def chat_qa(self,query):
      results = self.qa_search(query)
      system = {"role":"system","content":"You are a helpful chatbot that answers questions based on data in your fact database."}
      messages = [system]
      text = f"Question: {query}\n\nFact Data: \n{results}"
      messages.append({"role":"user","content":text})
      output = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=messages)["choices"][0]["message"].to_dict()["content"]
      return output
    
    def clean_graph(self,key):
      facts = [key+": "+str(rel).replace("description","is")+" "+str(self.entities[key][rel]) for rel in self.entities[key]]
      rels = [rel for rel in self.entities[key]]
      fact_embs = self.model.encode(facts)
      scores = util.cos_sim(fact_embs,fact_embs)
      pairs = []
      for i in range(len(scores)):
        for j in range(len(scores[i])):
          if round(scores[i][j].item(),3)!=1.0 and scores[i][j]>0.7:
            if (facts[i],facts[j]) not in pairs and (facts[j],facts[i]) not in pairs:
              pairs.append((facts[i],facts[j]))
      for pair in pairs:
        system = {"role":"system","content":"You are a helpful chatbot that only outputs YES or NO"}
        messages = [system]
        messages.append({"role":"user","content":f"Do these two facts in our database express the same thing?: {pair}"})
        output = openai.ChatCompletion.create(model="gpt-4",messages=messages)["choices"][0]["message"].to_dict()["content"]
        if "yes" in output.lower():
          bad_index = facts.index(pair[1])
          redundant = rels[bad_index]
          del self.entities[key][redundant]
          good_index = facts.index(pair[0])
          validated = rels[bad_index]
          try:
            self.fact_scores[(key,validated)]+=1
          except:
            self.fact_scores[(key,validated)]=1

    def load_graph(self,kg_file):
      with open(kg_file) as f:
        lines = f.readlines()
        graph_data = "\n".join(lines[:-1])
        ents = eval(lines[-1])
      data = [{ent:[{key:ents[ent][key]} for key in ents[ent]]} for ent in ents]
      json_string = json.dumps(data)
      print(json_string)
      self.update_graph(str(json_string)) 
      self.graph = pydot.graph_from_dot_data(graph_data)[0]

    def save_graph(self,filename="mygraph.kg"):
      with open(filename,"w") as f:
        f.write("")
      self.graph.write_dot(filename)
      with open(filename,"a") as f:
        f.write("\n")
        f.write(str(self.entities))
