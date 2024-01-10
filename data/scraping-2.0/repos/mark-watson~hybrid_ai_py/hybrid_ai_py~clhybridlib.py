## Semantic Web and Linked Data ##

import rdflib
from SPARQLWrapper import SPARQLWrapper
from pprint import pprint

def query_helper(sparql_query, endpoint):
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat('json')
    r = sparql.query().convert()
    vars = r['head']['vars']
    results = r['results']['bindings']
    rr = [vars]
    for r1 in results:
        rr.append([r1[v]['value'] for v in vars])
    return rr

def query_dbpedia(sparql_query):
    return query_helper(sparql_query, 'http://dbpedia.org/sparql')


## OpenAI GPT-3 APIs ##

import os
import openai

# Load API key from an environment variable
openai.api_key = os.getenv("OPENAI_KEY")


def generate_text(prompt, temperature=0.7, top_p=0.9, max_tokens=50):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": prompt},
      ]
    ) 
    print("**response:", response)  
    return response['choices'][0]['message']['content']

## My BERT + DBPedia QA ##

from transformers import pipeline

qa = pipeline(
    "question-answering",
    #model="NeuML/bert-small-cord19qa",
    model="NeuML/bert-small-cord19-squad2",
    tokenizer="NeuML/bert-small-cord19qa"
)
import spacy

nlp_model = spacy.load('en_core_web_sm')

from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("http://dbpedia.org/sparql")

def query(query):
  sparql.setQuery(query)
  sparql.setReturnFormat(JSON)
  return sparql.query().convert()['results']['bindings']

def entities_in_text(s):
    doc = nlp_model(s)
    ret = {}
    for [ename, etype] in [[entity.text, entity.label_] for entity in doc.ents]:
        if etype in ret:
            ret[etype] = ret[etype] + [ename]
        else:
            ret[etype] = [ename]
    return ret


def dbpedia_get_entities_by_name(name, dbpedia_type):
  sparql = "select distinct ?s ?comment where {{ ?s <http://www.w3.org/2000/01/rdf-schema#label>  \"{}\"@en . ?s <http://www.w3.org/2000/01/rdf-schema#comment>  ?comment  . FILTER  (lang(?comment) = 'en') . ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> {} . }} limit 15".format(name, dbpedia_type)
  #print(sparql)
  results = query(sparql)
  return(results)

entity_type_to_type_uri = {'PERSON': '<http://dbpedia.org/ontology/Person>',
    'GPE': '<http://dbpedia.org/ontology/Place>', 'ORG':
    '<http://dbpedia.org/ontology/Organisation>'}

def QA(query_text):
  entities = entities_in_text(query_text)

  def helper(entity_type):
    ret = ""
    if entity_type in entities:
      for hname in entities[entity_type]:
        results = dbpedia_get_entities_by_name(hname, entity_type_to_type_uri[entity_type])
        for result in results:
          ret += ret + result['comment']['value'] + " . "
    return ret

  context_text = helper('PERSON') + helper('ORG') + helper('GPE')
  print("\ncontext text:\n", context_text, "\n")

  print("Answer from transformer model:")
  print("Original query: ", query_text)
  print("Answer:")

  answer = qa({
                "question": query_text,
                "context": context_text
               })
  print(answer['answer'])
  return answer['answer']
