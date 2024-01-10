import os
import openai
from tqdm import tqdm
from rdflib import Graph, URIRef, Namespace, Literal
from rdflib.namespace import SKOS, RDF, RDFS, XSD
import datetime
import text2term
from pyshacl import validate
import pandas as pd
from word_forms.word_forms import get_word_forms
import re
from itertools import chain
import spacy
import numpy as np
import pickle
#!python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")
import argparse

def semantify_paper_batch(papers,api_key,max=None):
    full_g = Graph()
    openai.api_key = api_key

    for paper in tqdm(papers[0:max]):

        prompt_text = """
        '"""+paper['abstract']+ """'

        Describe the claim of the abstract above only using RDF (the turtle syntax), and using the following ontology:

        :hasStudy rdfs:domain bibo:AcademicArticle .
        :hasStudy rdfs:range sio:ObservationalStudy .
        :hasHypothesis rdfs:domain sio:ObservationalStudy .
        :hasHypothesis rdfs:range sio:Hypothesis .
        :independentVariable rdfs:domain sio:Hypothesis.
        :independentVariable rdfs:range qb:DimensionProperty .
        :mediatorVariable rdfs:domain sio:Hypothesis .
        :mediatorVariable rdfs:range :DimensionProperty .
        :dependentVariable rdfs:domain sio:Hypothesis .
        :dependentVariable rdfs:range qb:MeasureProperty.
        :hasRelation rdfs:domain :Hypothesis .
        :hasRelation rdfs:range :RelationProperty .
        :hasQualifier rdfs:domain :Hypothesis .
        :hasQualifier rdfs:range :Qualifier .
        :moderatorVariable rdfs:domain sio:Hypothesis .
        :moderatorVariable rdfs:range :DimensionProperty .
        :moderatorEffectOnStatementStrength rdfs:domain :Hypothesis .
        :moderatorEffectOnStatementStrength rdfs:range :Qualifier .
        :moderatorContext rdfs:domain sio:Hypothesis . 
        :moderatorContext rdfs:range sio:HumanPopulation, sio:GeographicRegion, sio:Organization . 
        :hasContext rdfs:domain sio:Hypothesis, :Moderator, :Mediator .
        :hasContext rdfs:range sio:HumanPopulation, sio:GeographicRegion, sio:Organization .
        :representedBy rdfs:domain sio:HumanPopulation, sio:GeographicRegion, sio:Organization .
        :representedBy rdfs:range :Sample .
        time:hasTime rdfs:domain :Sample .
        time:hasTime rdfs:range time:TemporalEntity .
        sem:hasPlace rdfs:domain :Sample .
        sem:hasPlace rdfs:range sio:GeographicRegion .
        geonames:locatedIn rdfs:domain sio:GeographicRegion .
        geonames:locatedIn rdfs:range geonames:Feature .
        time:hasBeginning rdfs:domain rdf:TemporalEntity .
        time:hasBeginning rdfs:range time:Instant .
        time:hasEnd rdfs:domain rdf:TemporalEntity .
        time:hasEnd rdfs:range time:Instant .
        time:inXSDDate rdfs:domain time:Instant .
        time:inXSDDate rdfs:range rdf:XMLLiteral .

        1. use rdfs:label to describe all blank nodes, also the geographic region. Use short descriptions, pieces of text verbatim from the abstract, and add language tags to all labels. An example would be: [] :hasSubject [ rdfs:label 'social class'@en ].

        2. for instances of the class geonames:Feature, find the URI for the place name in geonames (uri = https://www.geonames.org/<code>) like so: [] geonames:locatedIn <uri> (Don't leave any spaces between the URI and angle brackets). If you cannot find a place in the abstract, omit these triples.

        3. use prefixes (xsd,time,ex,geonames,rdf,rdfs,sem,skos,sio,sp) (the namespace for sp is https://w3id.org/linkflows/superpattern/latest/)

        4. the individual of bibo:AcademicArticle is: ex:"""+paper['paperId']+""". Don't enclose the individual with brackets.

        5. If you can't find a time reference in the abstract, try to estimate the dates.

        6. include all classes of all individuals using rdf:type

        7. a hypothesis describes the effect of an independent variable (such as social class or age) on a dependent variable (such as mortality). Optional variables are: a mediating variable (such as a country's living standards), which explains the process through which the independent and dependent variables are related, and a moderating variable which affects the strength and direction of that relationship.

        8. for values of :hasRelation use sp:affects

        9. for qualifiers, choose from the following: :strongMediumNegative, :strongMedium, :weakNegative, :weak, :no, :weakPositive, :strongMediumPositive)

        10. don't create your own identifiers but use blank nodes in case no IRI is available

        11. make sure to add a qualifier for the relation between independent and dependent variable, but also to the moderator.

        12. link hypothesis contexts to study indicators of that context. For example, if the modifier variable is food prices, it could be that the context is a geographic region with the indicator Recession.

        Only return proper RDF, no free text comments.
        """

        try:
            response = openai.ChatCompletion.create(model="gpt-4",
                                                messages=[{"role": "user", "content": prompt_text}],
                                                temperature=0)
        except:
            response = openai.ChatCompletion.create(model="gpt-4",
                                                messages=[{"role": "user", "content": prompt_text}],
                                                temperature=0)
        prefixes = """
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        @prefix bibo: <http://purl.org/ontology/bibo/> .
        @prefix time: <http://www.w3.org/2006/time#> .
        @prefix ex: <http://example.org/> .
        @prefix geonames: <http://www.geonames.org/ontology#> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix sem: <http://semanticweb.cs.vu.nl/2009/11/sem/> .
        @prefix skos: <http://www.w3.org/2004/02/skos/core#> .
        @prefix sio: <http://semanticscience.org/resource/> .
        @prefix sp: <https://w3id.org/linkflows/superpattern/latest/> .
        @prefix qb: <http://purl.org/linked-data/cube#> .
        @prefix : <https://w3id.org/mira/> .
        """
        try:
            print(paper['abstract']+'\n')
            print(response.choices[0].message.content)
            g=Graph()
            g.parse(data=prefixes+response.choices[0].message.content, format="turtle")
            full_g += g
        except Exception as e:
            print(e)
    full_g = full_g.skolemize()
    return full_g

def process_graph(batch):
    # Define the old and new IRI prefixes
    old_prefixes = ['HumanPopulation','https://w3id.org/mira/hasStudy','Organization','GeographicRegion','GeographicalRegion',
                    'https://w3id.org/mira/representedBy','ObservationalStudy','https://w3id.org/mira/hasHypothesis',
                    'independentVariable','dependentVariable','http://semanticscience.org/resource/Hypothesis','moderatorVariable',
                    'mediatorVariable','https://w3id.org/mira/DimensionProperty',
                    'http://purl.org/linked-data/cube#/DimensionProperty','http://purl.org/linked-data/cube#/MeasureProperty','https://w3id.org/mira/Sample']
    new_prefixes = ['SIO_001062','http://semanticscience.org/resource/SIO_000008','SIO_000012','SIO_000414','SIO_000414',
                    'http://semanticscience.org/resource/SIO_000205','SIO_000976','http://semanticscience.org/resource/SIO_000008',
                    'hasSubject','hasObject','https://w3id.org/mira/Explanation','hasModerator','hasMediator',
                    'http://purl.org/linked-data/cube#DimensionProperty','http://purl.org/linked-data/cube#DimensionProperty',
                    'http://purl.org/linked-data/cube#MeasureProperty','http://semanticscience.org/resource/SIO_001050']

    # Iterate through the triples in the graph and replace IRIs
    new_triples = []
    for subject, predicate, obj in batch:
        for old_prefix,new_prefix in zip(old_prefixes,new_prefixes):
            if isinstance(subject, URIRef):
                subject = URIRef(str(subject).replace(old_prefix, new_prefix))
            if isinstance(predicate, URIRef):
                predicate = URIRef(str(predicate).replace(old_prefix, new_prefix))
            if isinstance(obj, URIRef):
                obj = URIRef(str(obj).replace(old_prefix, new_prefix))
        new_triples.append((subject, predicate, obj))

    # Clear the old triples and add the new ones
    batch = Graph()
    for triple in new_triples:
        batch.add(triple)

    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    prefix mira: <https://w3id.org/mira/>
    PREFIX sio: <http://semanticscience.org/resource/>

    construct {
        ?study sio:SIO_000008 ?interaction .  
         ?interaction a mira:Explanation ;
                 a mira:InteractionEffect ;
                 mira:hasSubject ?mod_var ;
                 mira:hasRelation mira:moderates ;
                 mira:hasObject ?exp ;
                 mira:hasQualifier ?qual ;
                 mira:hasContext ?context . 
         ?context sio:SIO_000205 ?sample . 
         ?mod_var ?p ?o .
    } where {
        ?study sio:SIO_000008 ?exp .  
        ?exp a mira:Explanation ;
            mira:hasModerator ?mod_var ;
            mira:moderatorEffectOnStatementStrength ?qual ;
            mira:moderatorContext ?context ;
            mira:hasContext/sio:SIO_000205 ?sample . 
        ?mod_var ?p ?o .
        ?mod_var rdfs:label ?label .
        BIND (IRI(CONCAT("https://w3id.org/mira/", REPLACE(LCASE(STR(?label)), " ", "_"))) AS ?interaction)
    }
    """
    # Execute the SPARQL query
    query_result = batch.query(query)

    mods = Graph()
    for row in query_result:
        s, p, o = row
        mods.add((s, p, o))

    delete_query = """
    prefix mira: <https://w3id.org/mira/>

    delete {?exp mira:hasModerator ?mod_var } where {?exp mira:hasModerator ?mod_var };

    """
    batch.update(delete_query)

    delete_query = """
    prefix mira: <https://w3id.org/mira/>

    delete {?exp mira:moderatorEffectOnStatementStrength ?qual } where {?exp mira:moderatorEffectOnStatementStrength ?qual }

    """
    batch.update(delete_query)
    
    delete_query = """
    prefix mira: <https://w3id.org/mira/>

    delete {?exp mira:moderatorContext ?context } where {?exp mira:moderatorContext ?context }

    """
    batch.update(delete_query)
    
    batch += mods
    return batch

def add_bibo_metadata(papers,batch):
    for s,p,o in batch.triples((None,RDF.type,URIRef("http://purl.org/ontology/bibo/AcademicArticle"))):
        for paper in papers:
            if paper['paperId'] == s.n3().split('/')[-1].split('>')[0]:
                batch.add((s,URIRef("http://purl.org/dc/terms/identifier"),Literal(paper['paperId'])))
                batch.add((s,URIRef("http://purl.org/dc/terms/title"),Literal(paper['title'])))
                batch.add((s,URIRef("http://purl.org/dc/terms/abstract"),Literal(paper.abstract)))
                doi = 'https://doi.org/'+paper['externalIds']['DOI']
                batch.add((s,URIRef("http://prismstandard.org/namespaces/1.2/basic/doi"),URIRef(doi)))
                if paper['publicationDate'] != None:
                    date = paper['publicationDate'].split('-')
                    date_obj = datetime.date(int(date[0]), int(date[1]), int(date[2]))
                else:
                    year = paper['year']
                    date_obj = datetime.date(year, 1, 1)
                date_str = date_obj.isoformat()
                date_literal = Literal(date_str, datatype=XSD.date)
                batch.add((s,URIRef("http://purl.org/dc/terms/created"),Literal(date_literal)))
                for author in paper.authors:
                    if author.authorId != None:
                        batch.add((s,URIRef("http://purl.org/dc/terms/contributor"),URIRef(author['url'])))
                for referenceId in [ref.paperId for ref in paper.references if ref.paperId != None]:
                    batch.add((s,URIRef("http://purl.org/ontology/bibo/cites"),URIRef('http://example.org/'+referenceId)))
    return batch

def add_geonames_metadata(batch):

    query = """
    PREFIX wgs84_pos: <http://www.w3.org/2003/01/geo/wgs84_pos#>
    prefix gn: <http://www.geonames.org/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    construct {
    ?location gn:locatedIn ?geonamesId .
    ?geonamesId rdf:type gn:Feature .
    ?geonamesId gn:name ?label .
    ?geonamesId wgs84_pos:long ?long .
    ?geonamesId wgs84_pos:lat ?lat .

    } where {
    ?location gn:locatedIn ?geoLocation .
    BIND(IRI(CONCAT(CONCAT("http://sws.geonames.org/",REPLACE(STR(?geoLocation),"https://www.geonames.org/", "")),"/")) AS ?geonamesId)

      SERVICE <http://factforge.net/repositories/ff-news> {
            ?geonamesId gn:name ?label .
            ?geonamesId wgs84_pos:long ?long .
            ?geonamesId wgs84_pos:lat ?lat .
            FILTER ( datatype(?lat) = xsd:float)
            FILTER ( datatype(?long) = xsd:float)
       }
    }
    """

    query_result = batch.query(query)

    geo = Graph()
    for row in query_result:
        s, p, o = row
        geo.add((s, p, o))

    delete_query = """
    prefix gn: <http://www.geonames.org/ontology#>
    delete {?location gn:locatedIn ?geoLocation } where {?location gn:locatedIn ?geoLocation }

    """
    batch.update(delete_query)

    batch += geo
    return batch

def validate_graph(batch,shacl_file):
    shacl_graph = Graph()
    shacl_graph.parse(shacl_file, format="ttl")
    print(shacl_graph.serialize(format='turtle'))
    r = validate(batch,
          shacl_graph=shacl_graph)
    conforms, results_graph, results_text = r

    if conforms:
        print("Validation successful. No violations found.")
    else:
        print("Validation failed. Violations found.")

        # Extract the violations from the results_graph
        violations = list(results_graph.triples((None, None, None)))

        # Print the number of violations
        print(results_graph.serialize(format='turtle'))

def get_variables(batch):
    query = """
        PREFIX qb: <http://purl.org/linked-data/cube#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        select ?concept ?label where {
            VALUES ?type {qb:DimensionProperty qb:MeasureProperty}.
            ?concept a ?type .
            ?concept rdfs:label ?label
        }
    """
    results = batch.query(query)
    data = []
    for row in results:
        data.append({'concept':row.concept, 'label':row.label })
    return pd.DataFrame(data)

def get_nes(var):
    # Process the text with SpaCy
    doc = nlp(var)

    # Extract named entities
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]

    # Retrieve noun phrases (nouns with their modifying adjectives)
    noun_phrases = []
    current_phrase = ""
    for token in doc:
        if token.pos_ == "NOUN" or token.pos_ == "ADJ":
            current_phrase += token.text + " "
        else:
            if current_phrase.strip() != "":
                noun_phrases.append(current_phrase.strip())
            current_phrase = ""
    return noun_phrases

def clean_terms_for_mapping(graph_df):
    for var in graph_df.columns.tolist():
        if var.endswith('label'):
            dic = dict()
            dic = {idx:get_nes(value.n3().split('@en')[0]) if value else '' for idx,value in enumerate(graph_df[var].values)}
            dic = {key: [item.lower().replace('inequality','') for item in value] for key, value in dic.items()}
            dic = {key: [item.lower().replace('population','') for item in value] for key, value in dic.items()}
            dic = {key: [item.lower().replace('composition','') for item in value] for key, value in dic.items()}
            dic = {key: [item.lower().replace('equality','') for item in value] for key, value in dic.items()}
            dic = {key: [value+re.split('/| ',item) for item in value] for key, value in dic.items()}
            dic = {key: list(chain.from_iterable(value)) for key, value in dic.items()}
            dic = {key: list(set(value+list(chain.from_iterable([list(get_word_forms(item,0.7)['n']) for item in value])))) for key, value in dic.items()}
            dic = {key: [item for item in value if item != ''] for key, value in dic.items()}
            graph_df[var+'_cleaned'] = dic
    return graph_df


def map_to_bioportal(df):
    mappings = pd.DataFrame(columns=["Source Term ID","Source Term","Mapped Term Label","Mapped Term CURIE","Mapped Term IRI","Mapping Score","Tags"])
    for idx,row in df.iterrows():
        try:
            mapping = text2term.map_terms(row.values[0],
                                   target_ontology='MESH,DOID,HHEAR,SIO,IOBC',
                                   min_score=0.9,
                                   separator=',',
                                   use_cache=True,
                                   term_type='classes',
                                    mapper="bioportal",
                                   incl_unmapped=False)

            mappings = pd.concat([mappings, mapping], ignore_index=True)
        except:
            pass
    return mappings


def retrieve_mappings(row_graph_df,mappings):
    superstring = ', '.join(row_graph_df[0])
    if superstring:
        return list(set([(row['Mapped Term IRI'],row['Mapped Term Label']) for idx,row in mappings.iterrows() if row['Source Term'].lower() in superstring.lower()]))
    else:
        return None

def annotate_graph_bioportal(batch):
    #get variables to map
    df = get_variables(batch)

    #clean variables (split terms, find variations)
    df = clean_terms_for_mapping(df)

    #find mappings for cleaned terms
    mappings = map_to_bioportal(df[['label_cleaned']])
    df['mappings'] = df[['label_cleaned']].apply(retrieve_mappings, axis=1,args=(mappings,))

    #add mappings to concepts
    bioIRIs = Graph()
    for idx,row in df.iterrows():
        if row.mappings:
            for identifier,label in row['mappings']:
                bioIRIs.add((URIRef(row['concept']),URIRef("http://purl.org/linked-data/cube#concept"),URIRef(identifier)))
                bioIRIs.add((URIRef(identifier),RDFS.label,Literal(label)))
                bioIRIs.add((URIRef(identifier),RDF.type,SKOS.Concept))
    batch += bioIRIs
    return batch



def main(paper_file, api_key, max, output, view=0):

    print("Loading data from file "+paper_file+"...")
    with open(paper_file, 'rb') as fp:
        papers = pickle.load(fp)
    if  view:
        print("OpenAI annotation of batch ...")
    batch = semantify_paper_batch(papers,api_key,max)
    if view:
        print(batch.serialize(format='turtle'))
    print("Process graph ...")
    batch = process_graph(batch)
    if view:
        print(batch.serialize(format='turtle'))
    print("Add bibliographic information ...")
    batch = add_bibo_metadata(papers,batch)
    if view:
        print(batch.serialize(format='turtle'))
    print("Link geonames metadata ...")
    batch = add_geonames_metadata(batch)
    if view:
        print(batch.serialize(format='turtle'))
    print("Link concepts to terms from BioPortal (can take longer) ...")
    batch = annotate_graph_bioportal(batch)
    if view:
        print(batch.serialize(format='turtle'))
    print("Store graph to file "+output+" ...")
    batch.serialize(output,format="ttl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script turning paper abstracts into RDF in terms of the MIRA ontology.")

    parser.add_argument("--paper_file", type=str, help="Path to the file with paper abstracts. File content has to be a list of dictionaries with the following keys: dict_keys(['paperId','title','abstract','year','publicationDate','authors','references'])")
    parser.add_argument("--api_key", type=str, help="Key for openai.api_key")
    parser.add_argument("--output", type=str, help="Path to .ttl file for storing the output graph.")
    parser.add_argument("--max", type=int, help="Max number of files to process.")
    parser.add_argument("--view", type=int, help="Print the annotations after each processing step, for debugging. Default 0")
    args = parser.parse_args()
    main(args.paper_file, args.api_key, args.max, args.output,args.view)
