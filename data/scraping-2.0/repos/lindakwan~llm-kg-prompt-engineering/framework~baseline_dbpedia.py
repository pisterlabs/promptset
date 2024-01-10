import os
import json
import csv
import re
import subprocess
import ast

import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from SPARQLWrapper import SPARQLWrapper, JSON
import spacy

import utilities.sparql_functions as sparql_f
import utilities.eval_metrics as eval_metrics
import utilities.llm_tasks_prompts as llm_tasks
from utilities.timeout import time_limit, TimeoutException

# sparql_wd = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql_dbp = SPARQLWrapper("http://dbpedia.org/sparql")

# Load the data
file = open("../data/mmlu_test/high_school_government_and_politics_test.csv", "r")
csv_reader = csv.reader(file, delimiter=',')
data = []
for row in csv_reader:
    data.append({"question_text": row[0], "choices": row[1:-1], "correct_answer": row[-1]})

# Create the language model
llm = OpenAI(temperature=0)

# Create NLP model for extracting entities
nlp = spacy.blank("en")
nlp.add_pipe("dbpedia_spotlight")

# Create a list of QA pairs
qa_pairs = dict()

num_correct = 0

# Generate a response for each question
for i, item in enumerate(data):
    question = item['question_text']
    response = llm.predict(question)

    print("Q:", question)
    print("A:", response.strip(), "\n")

    qa_pairs[i] = dict()
    qa_pairs[i]["question"] = question
    qa_pairs[i]["choices"] = item['choices']
    qa_pairs[i]["initial_response"] = response.strip()

    # dbp_spotlight_output = nlp(question + " " + response.strip())
    # ent_list = [(ent.text, ent.kb_id_, ent._.dbpedia_raw_result['@similarityScore']) for ent in
    #             dbp_spotlight_output.ents]
    # ent_ids = [ent.kb_id_ for ent in dbp_spotlight_output.ents]

    # print("Doc:", dbp_spotlight_output)
    # print("Entities:", ent_list)

    new_response = ""
    try:
        with time_limit(60):
            question_escaped = question.replace('"', '\\\"')
            response_escaped = response.strip().replace('"', '\\\"')

            entity_names = llm_tasks.extract_entities(f"{question} {response.strip()}")

            print(entity_names)
            num_of_identified_ents = len(entity_names)
            print("Number of entities identified:", num_of_identified_ents)
            print()

            qa_pairs[i]["entity_names"] = entity_names

            # Extract entities and relations from the response
            # Get top 5 results
            print("Retriving from FALCON...")
            falcon_output = subprocess.run(["curl", "--header", "Content-Type: application/json", "--request", "POST",
                                            "--data", f"{{\"text\":\"{question_escaped} {response_escaped}\"}}",
                                            'https://labs.tib.eu/falcon/falcon2/api?mode=long&db=1&k=3'], capture_output=True, text=True)
            print("Retrival complete.\n")

            # Obtain list of relations from extraction process
            try:
                dic_ents_rels = json.loads(falcon_output.stdout)
                relations_dbpedia = dic_ents_rels['relations_dbpedia']
                entities_dbpedia = dic_ents_rels['entities_dbpedia']
            except json.decoder.JSONDecodeError:
                dic_ents_rels = dict()
                relations_dbpedia = []
                entities_dbpedia = []

            entities_ids = [ent['URI'] for ent in entities_dbpedia]
            print("Entities:", entities_ids)
            print()

            qa_pairs[i]["entities_ids"] = entities_ids

            relations_ids = [rel['URI'] for rel in relations_dbpedia]
            print("Relations:", relations_ids)
            print()

            qa_pairs[i]["relations_ids"] = relations_ids

            # Feed question, LLM response, and entities and relations into LLM
            # Extract knowledge graph facts from the response
            triples = llm_tasks.extract_kg_facts(f"{question} {response.strip()}", entities_ids, relations_ids)
            print("Triples:", triples)
            print()

            qa_pairs[i]["triples"] = triples

            true_count = 0
            true_facts_uris = []
            true_facts_names = []

            # For each triple, perform a SPARQL query to verify the truthfulness
            for s, p, o in triples:
                # Convert the triple to SPARQL format
                subject, s_name = sparql_f.uri_to_sparql_format(s)
                predicate, p_name = sparql_f.uri_to_sparql_format(p)
                obj, o_name = sparql_f.uri_to_sparql_format(o)

                sparql_query = f"ASK {{{subject} {predicate} {obj}.}}"

                print(sparql_query)

                # Perform the SPARQL query
                sparql_result = sparql_f.execute_sparql_query(sparql_query, sparql_dbp)
                print("Result:", sparql_result["boolean"], "\n")
                print()

                if sparql_result["boolean"]:
                    true_count += 1
                    true_facts_uris.append((subject, predicate, obj))
                    true_facts_names.append((s_name, p_name, o_name))
                else:
                    # Swap subject and object in case if the direction is incorrect
                    sparql_query = f"ASK {{{obj} {predicate} {subject}.}}"
                    print(sparql_query)

                    # Perform the SPARQL query
                    sparql_result = sparql_f.execute_sparql_query(sparql_query, sparql_dbp)
                    print("Result:", sparql_result["boolean"], "\n")
                    print()

                    if sparql_result["boolean"]:
                        true_count += 1
                        true_facts_uris.append((obj, predicate, subject))
                        true_facts_names.append((o_name, p_name, s_name))

            print("True Count:", true_count)
            print("% True:", true_count / len(triples))
            print()

            qa_pairs[i]["true_count"] = true_count
            qa_pairs[i]["% true"] = true_count / len(triples)

            # Calculate the number of linked entities
            linked_entities = set()
            for s, p, o in triples:
                linked_entities.add(s)
                linked_entities.add(o)
            print("Linked Entities:", linked_entities)
            print("Number of Linked Entities:", len(linked_entities))

            # Evaluate the truthfulness of the response
            eval_score = eval_metrics.simple_evaluation(entity_names, linked_entities, triples, true_facts_uris)
            print("Evaluation Score:", eval_score)
            print()

            qa_pairs[i]["evaluation_score"] = eval_score

            # facts_sequence = ""
            #
            # for s_name, p_name, o_name in true_facts_names:
            #     facts_sequence += f"{s_name} {p_name} {o_name}. "
            #
            # print("Facts Sequence", facts_sequence)

        #     facts_seq_length = len(facts_sequence.strip().split(" "))
        #     response_length = len(response.strip().split(" "))
        #
        #     # Evaluate the truthfulness of the response
        #     print("Length Facts Sequence / Length Response:", facts_seq_length / response_length)
        #     print()

            if eval_score < 0.5:
                true_entities = dict()
                for j, (s_uri, _, o_uri) in enumerate(true_facts_uris):
                    s_name, _, o_name = true_facts_names[j]
                    true_entities[s_uri] = s_name
                    true_entities[o_uri] = o_name

                true_relations = dict()
                for j, (_, p_uri, _) in enumerate(true_facts_uris):
                    true_relations[p_uri] = true_facts_names[j][1]

                print("True entities:", true_entities)
                print("True relations:", true_relations)
                print()

                # Do knowledge graph enrichment
                filtered_facts = []
                if len(true_entities) > 0:
                    # Execute SPARQL query to get the list of predicate/object pairs
                    # subject = list(true_entities.keys())[0]
                    for subject in list(true_entities.keys()):
                        print("Subject:", subject)
                        sparql_query = f'SELECT ?predicate WHERE {{{subject} ?predicate ?object. \
                        FILTER(!isLiteral(?object) || lang(?object) = "" || langMatches(lang(?object), "EN"))}}'
                        print(sparql_query)
                        sparql_output = sparql_f.execute_sparql_query(sparql_query, sparql_dbp)
                        sparql_bindings = sparql_output['results']['bindings']

                        unique_predicates = dict()

                        for binding in sparql_bindings:
                            predicate = binding['predicate']['value']
                            predicate_alias = sparql_f.get_name_from_dbpedia_uri(predicate)
                            if predicate_alias not in unique_predicates:
                                unique_predicates[predicate_alias] = []
                            unique_predicates[predicate_alias].append(predicate)

                        # print("Unique predicates:", unique_predicates.keys())

                        # Given a list of predicates, use the LLM to get the order of predicates by most relevant
                        top_preds = llm_tasks.extract_relevant_predicates(question, list(unique_predicates.keys()), k=3)

                        print("Top predicates:", top_preds)
                        print()

                        # Execute SPARQL query for each of the top 5 predicates
                        for pred in top_preds:
                            pred_uri = unique_predicates[pred][0]
                            sparql_query = f'SELECT ?object WHERE {{{subject} <{pred_uri}> ?object. \
                            FILTER(!isLiteral(?object) || lang(?object) = "" || langMatches(lang(?object), "EN"))}}'
                            print(sparql_query)
                            sparql_output = sparql_f.execute_sparql_query(sparql_query, sparql_dbp)
                            sparql_bindings = sparql_output['results']['bindings']

                            # print(sparql_bindings)

                            for binding in sparql_bindings:
                                obj = binding['object']['value']
                                filtered_facts.append((subject, pred_uri, obj))

                        print()

                context_string = ""
                for s, p, o in filtered_facts:
                    s_name = true_entities[s]
                    p_name = sparql_f.get_name_from_dbpedia_uri(p)
                    o_name = sparql_f.get_name_from_dbpedia_uri(o)
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
            else:
                new_response = response.strip()
    except TimeoutException:
        print("Timeout Exception")
        new_response = response.strip()
        qa_pairs[i]["timeout"] = True
    except:
        new_response = response.strip()
        qa_pairs[i]["error"] = True

    mc_prompt = PromptTemplate(
        input_variables=["question", "response", "choices"],
        template="Output the best one of the numbered options for the following question and response:\n \
                        Question: {question}\nResponse: {response}\nOptions:\n{choices}"
    )

    choices_text = "\n".join([str(i + 1) + ". " + choice for i, choice in enumerate(item['choices'])])
    choice_response = llm(mc_prompt.format(question=question, response=new_response.strip(), choices=choices_text))

    print("Choice Response:", choice_response.strip())

    # Convert the response to the numbered choice
    numbers = [int(num) for num in re.findall(r'\d+', choice_response.strip().split(".")[0])]
    if len(numbers) == 0:
        numbered_output = 1
    else:
        numbered_output = numbers[-1]
    letter_output = chr(ord('A') + int(numbered_output) - 1)

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

    # new_response = llm.predict(f"{question}\nContext:{context_string}")



        # relevant_entities_json = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": "You will be provided with question text and a list of entities. \
        #             Your task is to order the entities by most relevant to text."
        #         },
        #         {
        #             "role": "user",
        #             "content": f"Text: {question}\nEntities: {unique_entities.keys()}"
        #         }
        #     ],
        #     temperature=0,
        #     max_tokens=256
        # )
        #
        # print(relevant_entities_json)

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

# Save the QA pairs in a JSON file
with open("../output/qa_sets_llm_kg_g&p01.json", "w") as f:
    json.dump(qa_pairs, f, indent=4)
