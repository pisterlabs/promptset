import openai
import os
import json
import pprint
import networkx as nx
import matplotlib.pyplot as plt

openai.api_key = os.getenv("OPENAI_API_KEY")
# mainTextPath = "Data/As I Lay Dying.txt"
mainTextPath = "Data\\As I Lay Dying.txt"
# mainText = open(mainTextPath, "r").read()
INDEX = "index"
NODES = "nodes"
EDGES = "edges"
nodesFile = "Nodes_NO_NM_WKG_3.csv"
edgesFile = "Edges_NO_NM_WKG_3.csv"
messageHistory = []

systemContent = f"""
Identify and list all the unique and important entities from the given text, make sure you don't miss anything.
From now on we call these entities as {NODES}.
The goal is to create a comprehensive knowledge graph of the text.
{NODES} are characters, relations, events, locations, objects, and concepts.
Node types must be mentioned in the output.
Do not include any Nodes that are not in the text.
Do not use prenouns or pronouns, use the full name of the Nodes. 
This is a Node sample: {{"id": "Ada", "T": "charachter"}}
You must also return the {NODES} in JSON format.
Moreover, show connections between the {NODES} as {EDGES} by using the following format:
{{"S": "Node1_id", "T": "Node2_id", "R": "Relation1", "ST": "positive", "key1": "daily rutine"}}
"S" is for source, "T" is for target, "R" is for relation, "ST" is for sentiment.
Relations are verbs, adjectives, and adverbs, none of them are unique names, all of them are general terms.
"""

systemMessage = {"role": 'system', "content": systemContent}
messageHistory.append(systemMessage)
def extract_entities(messages):
    try:
        response = openai.ChatCompletion.create(
        #   model='gpt-3.5-turbo',
          model='gpt-3.5-turbo-1106',
        #   model='gpt-4-1106-preview',
          messages=messages,
          temperature=1,
        #   max_tokens=1000,
          response_format= {'type': 'json_object'}
        )
        # Extract the entities from the response
        choices = response['choices']
        firstMessage = choices[0].message.content
        role = choices[0].message.role
        # print(role, "\n", firstMessage)
        return response

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

paragraphs = []
with open(mainTextPath, 'r', encoding='utf-8') as file:
    paragraph = []
    for line in file:
        if line.strip() == '':  # Check for blank lines
            if paragraph:
                paragraphs.append('\n'.join(paragraph))
                paragraph = []
        else:
            paragraph.append(line.strip())
    if paragraph:  # Add the last paragraph if the file doesn't end with a blank line
        paragraphs.append('\n'.join(paragraph))

paraCount = len(paragraphs)
paraphsPerMessage = 60
historyParaphsPerMessage = round(paraphsPerMessage * 0.1)
maxParaphParsed = 3000
nodes = dict()
edges = dict()
idx = 0
try:
    # for i in range(0, paraCount, paraphsPerMessage):
    i = 0
    while i < paraCount:
        if i >= maxParaphParsed or i >= paraCount:
            break
        print(f"sending paragraphs {i} to {i + paraphsPerMessage} out of {paraCount}")
        content = ""
        for j in range(i, i + paraphsPerMessage):
            if j >= paraCount:
                j = paraCount - 1
                content += paragraphs[j] + "\n"
                break
            content += paragraphs[j] + "\n"
        i += paraphsPerMessage
        existingKG = "This is the extracted data so far:\n"
        # existingKG += f"{NODES} = {json.dumps(list(nodes.values()))}\n{EDGES} = {json.dumps(list(edges.values()))}"
        existingKG += f"{NODES} = {json.dumps(list(nodes.values()))}\n"
        extractedKGMessage = {"role": 'user', "content": existingKG}
        newMessage = {"role": 'user', "content": content}
        messages = messageHistory + [extractedKGMessage] + [newMessage] 
        print("")
        print("sending request to openai")
        response = extract_entities(messages)
        # print number of tokens
        print(response['usage'])
        try:
            jsonRes = json.loads(response.choices[0].message.content)
            if NODES in jsonRes:
                nds = jsonRes[NODES]
                for nd in nds:
                    if nd["id"] not in nodes:
                        nodes[nd["id"]] = nd
            if EDGES in jsonRes:
                eds = jsonRes[EDGES]
                for ed in eds:
                    edges[idx] = ed
                    idx += 1
            # pprint.pprint(jsonRes)
            print(f"node count: {len(nodes)}, edge count: {len(edges)}")
        except Exception as e:
            # print(response)
            finish_reason = response['choices'][0]['finish_reason']
            if finish_reason == "length":
                i -= paraphsPerMessage // 2
            print("could not parse json")
            print(e)
except Exception as e:
    print("could not send request to openai")
    print(e)

import csv

with open(nodesFile, 'w', newline='') as csvfile: 
    fieldnames = ['id', 'T']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for nd in nodes:
        writer.writerow(nodes[nd])
with open(edgesFile, 'w', newline='') as csvfile:
    fieldnames = ['S', 'T', 'R', 'ST']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for ed in edges:
        writer.writerow(edges[ed])