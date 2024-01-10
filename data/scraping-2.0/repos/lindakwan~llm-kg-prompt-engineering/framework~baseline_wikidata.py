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
    dataset_path = "../data/mmlu_test/high_school_geography_test.csv"
    json_output_path = f"../output/qa_sets_llm_kg_geography_wd_{start_time.timestamp()}.json"
    # json_output_path = f"../output/qa_sets_llm_kg_geography01.json"
elif args.dataset == "government_and_politics":
    dataset_path = "../data/mmlu_test/high_school_government_and_politics_test.csv"
    json_output_path = f"../output/qa_sets_llm_kg_government_and_politics_wd_{start_time.timestamp()}.json"
    # json_output_path = f"../output/qa_sets_llm_kg_government_and_politics01.json"
else:
    print("Invalid dataset.")
    exit()

# Load the data
data = dio.read_data(dataset_path)

# Create the language model
llm = OpenAI(temperature=0)

num_correct = 0

# Generate a response for each question
for i, item in enumerate(data):  # 66:71  # 36:37
    question = item['question_text']
    response = llm.predict(question)

    print("Q:", question)
    print("A:", response.strip(), "\n")

    qa_pairs[i] = dict()
    qa_pairs[i]["question"] = question
    qa_pairs[i]["choices"] = item['choices']
    qa_pairs[i]["initial_response"] = response.strip()

    new_response = ""
    try:
        with time_limit(120):
            question_escaped = question.replace('"', '\\\"')
            response_escaped = response.strip().replace('"', '\\\"')

            # Use the LLM to extract entities from the response
            question_entities = llm_tasks.extract_entities(question)
            print("Q entities:", question_entities)

            response_entities = llm_tasks.extract_entities(response.strip())
            print("A entities:", response_entities)

            entity_names = list(dict.fromkeys(question_entities + response_entities))  # llm_tasks.extract_entities(f"{question} {response.strip()}")
            print("All entities:", entity_names)

            num_of_identified_ents = len(entity_names)
            print("Number of entities identified:", num_of_identified_ents)
            print()

            qa_pairs[i]["question_entity_names"] = question_entities
            qa_pairs[i]["response_entity_names"] = response_entities

            # Feed question, LLM response, and entities and relations into LLM
            # Extract knowledge graph facts from the response
            triples_names = llm_tasks.extract_kg_facts_given_entities(f"{question} {response.strip()}", entity_names)
            print("Triples:", triples_names)
            print()

            qa_pairs[i]["extracted_triples"] = triples_names

            entities_name_uri_map = dict()
            relations_name_uri_map = dict()

            uri_name_map = dict()

            extr_triples_uris = []

            # Convert the triples to a list of URIs
            for s, p, o in triples_names:
                # Use REST API to get the URI of the entity/property
                if s not in entities_name_uri_map:
                    entities_name_uri_map[s] = el.fetch_wikidata_from_query(s)
                if p not in relations_name_uri_map:
                    relations_name_uri_map[p] = el.fetch_wikidata_from_query(p, ent_type='property')
                if o not in entities_name_uri_map:
                    entities_name_uri_map[o] = el.fetch_wikidata_from_query(o)

                uris = []  # Used to construct the triple as a tuple of URIs

                for j, component in enumerate([s, p, o]):
                    # Retrieve the URI of the entity/property
                    if j == 1:
                        info = relations_name_uri_map[component]
                    else:
                        info = entities_name_uri_map[component]

                    if len(info['search']) == 0:
                        print('Sorry, no results for "' + component + '"')
                        uris.append('"' + component + '"')  # Use name of entity/property (quoted) instead
                        uri_name_map['"' + component + '"'] = component
                    else:
                        label = info['search'][0]["label"]
                        uri = info['search'][0]["concepturi"]
                        description = info['search'][0].get("description", "No description available.")
                        print(label, uri, description)
                        uris.append(uri)
                        uri_name_map[uri] = component
                print()

                extr_triples_uris.append(tuple(uris))

            # Print triples as tuple of URIs
            print("Extracted triples URIs:", extr_triples_uris)
            print()

            qa_pairs[i]["extracted_triples_uris"] = extr_triples_uris

            question_ent_uris = []
            for ent_name in question_entities:
                ent_uri = el.fetch_uri_wikidata(ent_name)
                question_ent_uris.append(ent_uri)
                uri_name_map[ent_uri] = ent_name

            qa_pairs[i]["question_entities_uris"] = question_ent_uris

            print("Question entities URIs:", question_ent_uris)
            print()

            qa_pairs[i]["uri_name_map"] = uri_name_map

            true_count = 0
            true_facts_uris = []
            true_facts_names = []
            true_entities_uris = set()

            # For each triple, perform a SPARQL query to verify the truthfulness
            for s, p, o in extr_triples_uris:
                # Convert the triple to SPARQL format
                s_format = sparql_f.uri_to_sparql_format_wikidata(s)
                p_format = sparql_f.uri_to_sparql_format_wikidata(p)
                o_format = sparql_f.uri_to_sparql_format_wikidata(o)

                # sparql_query = f"ASK {{{subject} {predicate} {obj}.}}"
                sparql_query = f"ASK {{{s_format} ?predicate {o_format}.}}"

                print(sparql_query)

                # Perform the SPARQL query
                sparql_result = sparql_f.execute_sparql_query(sparql_query, sparql_wd)
                print("Result:", sparql_result["boolean"], "\n")
                print()

                if sparql_result["boolean"]:
                    true_count += 1
                    true_facts_uris.append((s, p, o))
                    true_facts_names.append((uri_name_map[s], uri_name_map[p], uri_name_map[o]))
                    true_entities_uris.add(s)
                    true_entities_uris.add(o)

                else:
                    # Swap subject and object in case if the direction is incorrect
                    # sparql_query = f"ASK {{{obj} {predicate} {subject}.}}"
                    sparql_query = f"ASK {{{o_format} ?predicate {s_format}.}}"
                    print(sparql_query)

                    # Perform the SPARQL query
                    sparql_result = sparql_f.execute_sparql_query(sparql_query, sparql_wd)
                    print("Result:", sparql_result["boolean"], "\n")
                    print()

                    if sparql_result["boolean"]:
                        true_count += 1
                        true_facts_uris.append((o, p, s))
                        true_facts_names.append((uri_name_map[o], uri_name_map[p], uri_name_map[s]))
                        true_entities_uris.add(s)
                        true_entities_uris.add(o)

            print("True Count:", true_count)

            if len(extr_triples_uris) > 0:
                frac_true = true_count / len(extr_triples_uris)
            else:
                frac_true = 0

            print("% True:", frac_true)
            print()

            print("True facts names:", true_facts_names)

            qa_pairs[i]["true_count"] = true_count
            qa_pairs[i]["% true"] = frac_true

            # Calculate the number of linked entities
            linked_entities = set()
            for s, p, o in extr_triples_uris:
                linked_entities.add(s)
                linked_entities.add(o)
            print("Linked Entities:", linked_entities)
            print("Number of Linked Entities:", len(linked_entities))

            # Evaluate the truthfulness of the response
            eval_score = eval_metrics.simple_evaluation(entity_names, linked_entities,
                                                        extr_triples_uris, true_facts_uris)
            print("Evaluation Score:", eval_score)
            print()

            qa_pairs[i]["evaluation_score"] = eval_score

            if eval_score < 0.5:
                qa_pairs[i]["below_threshold"] = True

                print("True entities:", true_entities_uris)
                # print("True relations:", true_relations)
                print()

                qa_pairs[i]["true_entities"] = list(true_entities_uris)

                # Do knowledge graph enrichment
                filtered_facts = []

                # Combine true entities with entities extracted from question
                focus_entities = true_entities_uris.union(question_ent_uris)

                if len(focus_entities) > 0:
                    # Execute SPARQL query to get the list of predicate/object pairs for each subject
                    for subject in list(focus_entities):
                        s_format = sparql_f.uri_to_sparql_format_wikidata(subject)
                        print("Subject:", subject)

                        sparql_query = f'SELECT ?predicate ?propLabel ?object ?objectLabel WHERE {{ \
                        {s_format} ?predicate ?object. \
                        SERVICE wikibase:label {{ \
                            bd:serviceParam wikibase:language "en" . }} \
                        ?prop wikibase:directClaim ?predicate . \
                        ?prop rdfs:label ?propLabel.  filter(lang(?propLabel) = "en"). \
                        FILTER(!isLiteral(?object) || lang(?object) = "" || langMatches(lang(?object), "EN")) }}'

                        print(sparql_query)
                        sparql_output = sparql_f.execute_sparql_query(sparql_query, sparql_wd)
                        sparql_bindings = sparql_output['results']['bindings']

                        # print("SPARQL bindings:", sparql_bindings)

                        unique_predicates = dict()  # Map URI to alias

                        # Get uris and names of unique predicates
                        for binding in sparql_bindings:
                            predicate_uri = binding['predicate']['value']
                            predicate_name = binding['propLabel']['value']
                            if predicate_name not in unique_predicates:
                                unique_predicates[predicate_name] = []
                            if predicate_uri not in unique_predicates[predicate_name]:
                                unique_predicates[predicate_name].append(predicate_uri)

                        # print("Unique predicates:", unique_predicates)
                        # print()

                        # qa_pairs[i]["unique_predicates_" + uri_name_map[subject]] = unique_predicates

                        # Given a list of predicates, use the LLM to get the order of predicates by most relevant
                        top_preds = llm_tasks.extract_relevant_predicates(question, list(unique_predicates.keys()), k=3)

                        print("Top predicates:", top_preds)
                        print()

                        qa_pairs[i]["top_predicates_" + uri_name_map[subject]] = top_preds

                        # Execute SPARQL query for each of the top 3 predicates
                        for top_pred in top_preds:
                            pred_uri = unique_predicates[top_pred][0]
                            top_p_format = sparql_f.uri_to_sparql_format_wikidata(pred_uri)
                            # TODO: Use the existing SPARQL binding instead of executing a new SPARQL query
                            sparql_query = f'SELECT ?object ?objectLabel WHERE \
                            {{{s_format} {top_p_format} ?object. \
                            SERVICE wikibase:label {{ \
                                bd:serviceParam wikibase:language "en" . }} \
                            FILTER(!isLiteral(?object) || lang(?object) = "" || langMatches(lang(?object), "EN"))}}'
                            print(sparql_query)
                            sparql_output = sparql_f.execute_sparql_query(sparql_query, sparql_wd)
                            sparql_bindings = sparql_output['results']['bindings']

                            for binding in sparql_bindings:
                                objLabel = binding['objectLabel']['value']
                                filtered_facts.append((uri_name_map[subject], top_pred, objLabel))

                print(filtered_facts)
                print()

                qa_pairs[i]["filtered_facts_for_context"] = filtered_facts

                context_string = ""
                for s_name, p_name, o_name in filtered_facts:
                    context_string += f"{s_name} {p_name} {o_name}. "

                print("Context String:", context_string)

                qa_pairs[i]["context_string"] = context_string

                new_prompt = PromptTemplate(
                    input_variables=["question", "context"],
                    template="Question: {question}\nContext: {context}",
                )

                chain = LLMChain(llm=llm, prompt=new_prompt)
                new_response = chain.run({"question": question, "context": context_string})

                print("New Response:", new_response.strip())

                qa_pairs[i]["new_response"] = new_response.strip()
            else:
                qa_pairs[i]["below_threshold"] = False
                new_response = response.strip()
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

print("EM:", num_correct / len(data))

qa_pairs['finish_time'] = str(datetime.datetime.now())

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
