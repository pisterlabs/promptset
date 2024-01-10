import json
import argparse
import datetime

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from SPARQLWrapper import SPARQLWrapper

import utilities.sparql_functions as sparql_f
import utilities.eval_metrics as eval_metrics
import utilities.llm_tasks_prompts as llm_tasks
import utilities.entity_link as el
import utilities.data_io as dio
import utilities.nlp_tasks as nlp_tasks
import utilities.emb_tasks as emb_tasks
from utilities.timeout import time_limit, TimeoutException

# Create a list of QA pairs
qa_pairs = dict()

start_time = datetime.datetime.now()

qa_pairs['start_time'] = str(start_time)

sparql_wd = SPARQLWrapper("https://query.wikidata.org/sparql")
# sparql_dbp = SPARQLWrapper("http://dbpedia.org/sparql")

# Add arguments to the command line
parser = argparse.ArgumentParser(description='Run the combined LLM-KG on a dataset.')
parser.add_argument('-d', '--dataset', help='The dataset to run the LLM-KG on.', required=True)
args = parser.parse_args()

dataset_path = ""
json_output_path = ""

if args.dataset == "geography":
    dataset_path = "../data/mmlu_test/high_school_geography_test_filtered.csv"
    json_output_path = f"../output/qa_sets_llm_kg_geography_wd_{start_time.timestamp()}.json"
    # json_output_path = f"../output/qa_sets_llm_kg_geography01.json"
elif args.dataset == "government_and_politics":
    dataset_path = "../data/mmlu_test/high_school_government_and_politics_test_filtered.csv"
    json_output_path = f"../output/qa_sets_llm_kg_government_and_politics_wd_{start_time.timestamp()}.json"
    # json_output_path = f"../output/qa_sets_llm_kg_government_and_politics01.json"
elif args.dataset == "miscellaneous":
    dataset_path = "../data/mmlu_test/miscellaneous_test_filtered.csv"
    json_output_path = f"../output/qa_sets_llm_kg_miscellaneous_wd_{start_time.timestamp()}.json"
else:
    print("Invalid dataset.")
    exit()

# Load the data
data = dio.read_data(dataset_path)

num_correct = 0

# Generate a response for each question
for i, item in enumerate(data):  # 41:42
    question = item['question_text']
    response = llm_tasks.generate_response_weaker(question)  # TODO: Change to weaker model to produce the initial answer

    print("Q:", question)
    print("A:", response.strip(), "\n")

    qa_pairs[i] = dict()
    qa_pairs[i]["question"] = question
    qa_pairs[i]["choices"] = item['choices']
    qa_pairs[i]["initial_response"] = response.strip()

    new_response = ""
    try:
        with time_limit(480):
            # Use the LLM to extract entities from the question
            question_entities = llm_tasks.extract_entities(question)
            print("Q entities:", question_entities)

            # Use the LLM to extract entities from the response
            response_entities = llm_tasks.extract_entities(response.strip())
            print("A entities:", response_entities)

            # Combine the entities from the question and response
            entity_names = list(dict.fromkeys(question_entities + response_entities))
            print("All entities:", entity_names)

            qa_pairs[i]["question_entity_names"] = question_entities
            qa_pairs[i]["response_entity_names"] = response_entities

            name_uri_map = dict()
            uri_name_map = dict()
            uri_label_desc_map = dict()

            # Get the uris, label and description of the question and response entities
            for ent_name in entity_names:
                uri_label_desc = el.fetch_uri_or_none_wikidata(ent_name, question, ent_type='item')
                if uri_label_desc is not None:
                    uri, label, desc = uri_label_desc
                    name_uri_map[(ent_name, 'ent')] = uri
                    uri_name_map[uri] = label
                    uri_label_desc_map[uri] = (label, desc)
                    print(ent_name, uri, label, desc, '\n')
                else:
                    name_uri_map[(ent_name, 'ent')] = None
                    print("No URI found for", ent_name, '\n')

            qa_pairs[i]["uri_name_map"] = uri_name_map
            qa_pairs[i]["uri_label_desc_map"] = uri_label_desc_map

            results = sparql_f.perform_sparql_query_multiple_entities(uri_name_map.keys())

            # Create a named_triples dictionary with named triples as keys and the SPARQL results as values
            named_triples = dict()
            for result in results:
                if not result['oLabel']['value'].startswith("http://"):
                    named_triple = (result['sLabel']['value'], result['propLabel']['value'], result['oLabel']['value'])
                    named_triples[named_triple] = result

            # Select the most relevant triples to the question
            num_relevant_facts = max(len(uri_name_map), 3)
            named_triples_keys = list(named_triples.keys())  # Initialise the list of relevant facts to all the facts

            round_num = 1
            batch_size = 50
            relevant_facts = []
            # Do batch splitting
            for j in range(0, len(named_triples_keys), batch_size):
                sublist = list(named_triples_keys)[j:j+batch_size]
                relevant_facts = llm_tasks.extract_relevant_facts(question, relevant_facts + sublist,
                                                                  k=num_relevant_facts)
                print(f"Round {round_num}: {len(relevant_facts)} facts extracted")
                round_num += 1

            print("Relevant facts:", relevant_facts)

            qa_pairs[i]["relevant_facts"] = relevant_facts

            # Convert relevant facts to string representation
            relevant_facts_str = ""
            for fact in relevant_facts:
                relevant_facts_str += f"{fact[0]} {fact[1]} {fact[2]}. "

            qa_pairs[i]["relevant_facts_str"] = relevant_facts_str

            # Measure similarity between the response and the relevant facts
            eval_score = emb_tasks.calculate_squared_cos_sim(response.strip(), relevant_facts_str)
            print("Similarity score:", eval_score)

            qa_pairs[i]["evaluation_score"] = eval_score

            if eval_score < 0.75:
                # Expand the set of entities to include the entities in the relevant facts
                entities_expanded_uris = dict()

                for fact in relevant_facts:
                    if fact in named_triples:
                        entities_expanded_uris[fact[0]] = named_triples[fact]['s']['value']
                        if named_triples[fact]['o']['type'] == 'uri':
                            entities_expanded_uris[fact[2]] = named_triples[fact]['o']['value']

                for ent_name in question_entities:
                    if name_uri_map[(ent_name, 'ent')] is not None:
                        entities_expanded_uris[ent_name] = name_uri_map[(ent_name, 'ent')]

                print("Entities expanded:", entities_expanded_uris)

                # Perform SQL query on the expanded set of entities
                expanded_results = sparql_f.perform_sparql_query_multiple_entities(entities_expanded_uris.values())

                # Create an expanded named_triples dictionary with named triples as keys and SPARQL results as values
                expanded_named_triples = dict()
                for result in expanded_results:
                    if not result['oLabel']['value'].startswith("http://"):
                        expanded_named_triple = (result['sLabel']['value'], result['propLabel']['value'], result['oLabel']['value'])
                        expanded_named_triples[expanded_named_triple] = result

                # Select the most relevant triples to the question from the expanded KG
                num_relevant_facts = max(len(uri_name_map), 3) * 2
                expanded_named_triples_keys = list(expanded_named_triples.keys())

                round_num = 1
                batch_size = 50
                expanded_rel_facts = []
                # Do batch splitting
                for j in range(0, len(expanded_named_triples_keys), batch_size):
                    sublist = list(expanded_named_triples_keys)[j:j+batch_size]
                    expanded_rel_facts = llm_tasks.extract_relevant_facts(question, expanded_rel_facts + sublist,
                                                                          k=num_relevant_facts)
                    print(f"Round {round_num}: {len(expanded_rel_facts)} facts extracted")
                    round_num += 1

                print("Expanded relevant facts:", expanded_rel_facts)

                qa_pairs[i]["expanded_relevant_facts"] = expanded_rel_facts

                # Convert expanded relevant facts to string representation
                expanded_rel_facts_str = ""
                for fact in expanded_rel_facts:
                    if fact not in relevant_facts:  # Remove duplicates
                        expanded_rel_facts_str += f"{fact[0]} {fact[1]} {fact[2]}. "

                qa_pairs[i]["expanded_relevant_facts_str"] = expanded_rel_facts_str

                # Run the LLM on the question again, but with the context in front of it
                new_response = llm_tasks.generate_response_using_context_weaker(question, f"{relevant_facts_str}\n{expanded_rel_facts_str}")

                print("New Response:", new_response.strip())

                qa_pairs[i]["new_response"] = new_response.strip()

            else:
                qa_pairs[i]["below_threshold"] = False
                new_response = response.strip()

                # TODO: Use weaker LLM to generate response
    except TimeoutException:
        print("Timeout Exception")
        new_response = response.strip()
        qa_pairs[i]["timeout"] = True
    except Exception as exc:
        print("Error:", exc)
        new_response = response.strip()
        qa_pairs[i]["error"] = str(exc)

    # Generate the letter output based on the response
    letter_output = llm_tasks.select_mc_response_based(question, new_response.strip(), item['choices'])

    print("Generated answer:", letter_output)

    # Evaluate the response
    is_correct = letter_output == item['correct_answer']
    print("Correct answer:", item['correct_answer'])
    print("Correct:", is_correct, "\n")

    # Update the number of correct answers
    if letter_output == item['correct_answer']:
        num_correct += 1

    qa_pairs[i]["llm_answer"] = letter_output
    qa_pairs[i]["correct_answer"] = item["correct_answer"]
    qa_pairs[i]["llm_is_correct"] = is_correct

    with open(json_output_path, "w") as f:
        json.dump(qa_pairs, f, indent=4)

    '''
    # Example SPARQL Query
    SELECT ?subject ?predicate ?object
    WHERE
    {
        wd: Q1299 ?predicate ?object.
    }

    SELECT ?predicate ?object
    WHERE {
        <http://dbpedia.org/resource/Romance_languages> ?predicate ?object.
        FILTER((!isLiteral(?object) || lang(?object) = "" || langMatches(lang(?object), "EN"))
        && (regex(str(?predicate), "http://dbpedia.org/property/")
        || regex(str(?predicate), "http://dbpedia.org/ontology/")))
    }
    '''

em = num_correct / len(data)
print("EM:", em)

qa_pairs['finish_time'] = str(datetime.datetime.now())
qa_pairs['EM'] = em

# Save the QA pairs in a JSON file
with open(json_output_path, "w") as f:
    json.dump(qa_pairs, f, indent=4)

# 2 steps
# ASK {<http://www.wikidata.org/entity/Q652> ?predicate1 ?object1.
#      ?subject2 ?predicate2 <http://www.wikidata.org/entity/Q19814>.}

# Get predicate label
# SELECT ?predicate ?propLabel ?object ?objectLabel WHERE {
#   <http://www.wikidata.org/entity/Q43473> ?predicate ?object.
#   SERVICE wikibase:label {
#     bd:serviceParam wikibase:language "en" .
#   }
#   ?prop wikibase:directClaim ?predicate .
#   ?prop rdfs:label ?propLabel.  filter(lang(?propLabel) = "en").
#   FILTER(!isLiteral(?object) || lang(?object) = "" || langMatches(lang(?object), "EN"))
# }

# Get description
# SELECT ?p ?pLabel ?pDescription ?w ?wLabel ?wDescription WHERE {
#    wd:Q30 p:P6/ps:P6 ?p .
#    ?p wdt:P26 ?w .
#    SERVICE wikibase:label {
#     bd:serviceParam wikibase:language "en" .
#    }
# }

"""Query using multiple subjects
SELECT ?s ?sLabel ?p ?prop ?propLabel ?o ?oLabel WHERE {
  VALUES ?s { wd:Q3640 wd:Q43 wd:Q1362 wd:Q843 }.
  SERVICE <https://query.wikidata.org/sparql> {
    ?s ?p ?o .
    ?prop wikibase:directClaim ?p .
    ?prop rdfs:label ?propLabel.  FILTER(lang(?propLabel) = "en").
  }.
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en" .
  }.
  FILTER(!isLiteral(?o) || lang(?o) = "" || langMatches(lang(?o), "EN")) .
}"""
