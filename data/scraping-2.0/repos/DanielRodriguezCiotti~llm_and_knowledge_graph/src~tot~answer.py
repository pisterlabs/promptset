import openai
from prompt import *
import pickle
from neo4j import GraphDatabase
from dotenv import dotenv_values
import json

config = dotenv_values()


def run_query(driver, query):
    results, summary, keys = driver.execute_query(query, database="neo4j")
    return results


def results_to_list(results, filter=None):
    clean = []
    for result in results:
        node = list(result["t"].values())[0]
        relation = result["r"].type
        if filter:
            if relation == filter:
                clean.append((node, relation))
        else:
            clean.append((node, relation))
    return clean


def get_nodes(label, name, driver, filter=None):
    query = "MATCH (titleNode:{} {{name: '{}'}})-[r]->(t) RETURN r, t".format(label, name)
    results = run_query(driver, query)
    results = [edge[0] for edge in results_to_list(results, filter)]
    return results


def get_edges(label, name, driver, filter=None):
    query = "MATCH (titleNode:{} {{name: '{}'}})-[r]->(t) RETURN r, t".format(label, name)
    results = run_query(driver, query)
    results = results_to_list(results, filter)
    results = list(set(results))
    return results


def get_title(driver, label="title"):
    query = "MATCH (n:title) RETURN n;"
    results = run_query(driver, query)
    title = list(results[0]["n"].values())[0]
    return title


def parse_line(line):
    line = line.split(": ")
    if len(line) != 2:
        return "", 0
    score = float(line[1])
    section = line[0][2:]
    return section, score


def get_max_score(completion):
    content = completion["choices"][0]["message"]["content"]
    lines = content.split("\n")
    max_score, max_section = 0, ""
    for j, line in enumerate(lines):
        section, score = parse_line(line)
        if score > max_score:
            max_score = score
            max_section = section
    return max_section, max_score


def concepts_to_informations(concepts, driver):
    informations = []
    for concept in concepts:
        infos = get_edges("concept", concept, driver)
        for info in infos:
            info = [concept, " ".join(info[1].split("_")), info[0]]
            info = " ".join(info)
            informations.append(info)
    informations = ", ".join(informations)
    return informations


# strategy: 1 = get candidates with score over a threshold, 2 = get k candidates
def main(question, strategy=1, k=None):
    driver = GraphDatabase.driver(config["NEO4J_URL"], auth=(config["NEO4J_USER"], config["NEO4J_PASSWORD"]))
    openai.api_key = config["OPENAI_API_KEY"]
    title = get_title(driver, label="title")
    sections = get_nodes("title", title, driver, filter="section")
    prompt = question + scoring_prompt + ", ".join(sections)
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": prompt}
        ]
    )
    with open("answer_completion_0.json", "w") as f:
        json.dump(completion, f, indent=4)

    section, section_score = get_max_score(completion)
    subsections = get_nodes("section", section, driver, filter="subsection")
    if subsections == []:
        concepts = get_nodes("section", section, driver)
    else:
        prompt = question + scoring_prompt + ", ".join(subsections)
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": prompt}
            ]
        )
        with open("answer_completion_1.json", "w") as f:
            json.dump(completion, f, indent=4)
        subsection, subsection_score = get_max_score(completion)
        concepts = get_nodes("subsection", subsection, driver)
    informations = concepts_to_informations(concepts, driver)
    prompt = question + answering_prompt + informations
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": prompt}
        ]
    )
    with open("answer_completion_2.json", "w") as f:
        json.dump(completion, f, indent=4)
    with open("answer_completion.json", "w") as f:
        json.dump(completion, f, indent=4)

question = "what are the main concepts in the related work of contrastive learning?"

main(question)