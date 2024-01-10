# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from dotenv import load_dotenv
import os
from ai_models import pyGraphNetExplorer, llm_input_token_count
import networkx as nx
import jsonlines
import random
from mock_graph_data import generate_mock_graph
from networkx.readwrite import json_graph
from langchain.callbacks import get_openai_callback
import json
import re
import time
import sys
import numpy as np

# Load environ variables from .env, will not override existing environ variables
load_dotenv()

EACH_PROMPT_RUN_TIME = 1
OUTPUT_JSONL_PATH = 'logs/node10_log.jsonl'
GRAPH_PATH = "../data/graph_data/node10.json"

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return cb.total_tokens

def getGraphData():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    filename = GRAPH_PATH

    # Read the test_grpah json file and return the contents as json
    with open(filename, "r") as f:
        rawData = json.load(f)

    return rawData

def node_attributes_are_equal(node1_attrs, node2_attrs):
    # Check if "size", "color", and "labels" are equal for two nodes
    return (
        node1_attrs["size"] == node2_attrs["size"] and
        node1_attrs["color"] == node2_attrs["color"] and
        node1_attrs["labels"] == node2_attrs["labels"]
    )

def userQuery(prompt_list, graph_json):
    # Load the existing prompt and golden answers from Json
    golden_answer_filename = '../data/prompt_golden_ans.json'
    with open(golden_answer_filename, "r") as f:
        allAnswer = json.load(f)

    # for each prompt in the prompt_list, append it as the value of {'prompt': prompt}
    for each_prompt in prompt_list:
        graph_json = json.dumps(graph_json)

        with_graph_prompt = "Here is the original graph data: " + graph_json + ". Please answer the following question: " + each_prompt
        requestData = {'llm-prompt': with_graph_prompt, 'original-prompt': each_prompt}
        # print(requestData)
        prompt_accu = 0

        # Reset ret when it's a new test
        ret = None

        # Run each prompt for EACH_PROMPT_RUN_TIME times
        for i in range(EACH_PROMPT_RUN_TIME):
            if each_prompt not in allAnswer.keys():
                # terminate the code with error message
                raise SystemExit('Un-support ground truth for the current prompt.')

            print("Current prompt: ", each_prompt)
            print("Find the prompt in the list.")

            # Load the graph: if the graph is changed, it should be updated for the next input
            graph_filename = GRAPH_PATH
            # Read the test_grpah json file and return the contents as json
            with open(graph_filename, "r") as f:
                rawData = json.load(f)

            G = json_graph.node_link_graph(rawData)

            print("Calling model")
            answer = pyGraphNetExplorer.run(requestData['llm-prompt'])
            llm_output_token_count = count_tokens(pyGraphNetExplorer, answer)
            print("model returned")

            if answer.startswith("Answer:\n'''"):
                llm_answer = answer[12:-4]
                print("llm_answer: ", llm_answer)
            else:
                print("Answer: ", answer)
                raise SystemExit('Un-support model output format.')
            # Format the output to a json object
            json_string = '{' + llm_answer + '}'
            ret = json.loads(json_string)

            if ret['type'] == 'graph':
                if isinstance(ret['data'], nx.Graph):
                    # Create a nx.graph copy, so I can compare two nx.graph later directly
                    ret_graph_copy = ret['data']
                    jsonGraph = nx.node_link_data(ret['data'])
                    ret['data'] = jsonGraph
                    # Save the modified graph
                    if "Create a new graph" in requestData['llm-prompt']:
                        pass
                    else:
                        with open(graph_filename, "w") as f:
                            json.dump(jsonGraph, f)

                else:
                    # Convert the jsonGraph back to nx.graph, to check if they are identical later
                    ret_graph_copy = json_graph.node_link_graph(ret['data'])

            goldenAnswerCode = allAnswer[requestData['original-prompt']]

            # ground truth answer should already be checked to ensure it can run successfully
            exec(goldenAnswerCode)
            ground_truth_ret = eval("ground_truth_process_graph(G)")
            # if the type of ground_truth_ret is string, turn it into a json object
            if isinstance(ground_truth_ret, str):
                ground_truth_ret = json.loads(ground_truth_ret)

            # check type "text", "list", "table", "graph" separately.
            if ground_truth_ret['type'] == 'text' or ground_truth_ret['type'] == 'list':
                # if ret['data'] type is int, turn it into string
                if isinstance(ret['data'], int):
                    ret['data'] = str(ret['data'])
                if isinstance(ground_truth_ret['data'], int):
                    ground_truth_ret['data'] = str(ground_truth_ret['data'])

                if ground_truth_ret['data'] == ret['data']:
                    prompt_accu = ground_truth_check_accu(prompt_accu, requestData, ground_truth_ret, ret, llm_output_token_count)
                else:
                    ground_truth_check_debug(requestData, ground_truth_ret, ret, llm_output_token_count)

            elif ground_truth_ret['type'] == 'table':
                if ground_truth_ret['data'] == ret['data']:
                    prompt_accu = ground_truth_check_accu(prompt_accu, requestData, ground_truth_ret, ret, llm_output_token_count)
                else:
                    ground_truth_check_debug(requestData, ground_truth_ret, ret, llm_output_token_count)

            elif ground_truth_ret['type'] == 'graph':
                # Undirected graphs will be converted to a directed graph
                # with two directed edges for each undirected edge.
                ground_truth_graph = nx.Graph(ground_truth_ret['data'])
                # TODO: fix ret_graph_copy reference possible error, when it's not created.
                ret_graph = nx.Graph(ret_graph_copy)

                # Check if two graphs are identical, no weights considered
                if nx.is_isomorphic(ground_truth_graph, ret_graph, node_match=node_attributes_are_equal):
                    prompt_accu = ground_truth_check_accu(prompt_accu, requestData, ground_truth_ret, ret, llm_output_token_count)
                else:
                    ground_truth_check_debug(requestData, ground_truth_ret, ret, llm_output_token_count)

            # sleep for some time, to avoid the API call limit
            time.sleep(10)

        print("=========Current query process is done!=========")
        print(requestData['original-prompt'])
        print("Total test times: ", EACH_PROMPT_RUN_TIME)
        print("Testing accuracy: ", prompt_accu/EACH_PROMPT_RUN_TIME)

    return ret


def ground_truth_check_debug(requestData, ground_truth_ret, ret, llm_output_token_count):
    print("Fail the test, and here is more info: ")
    if ground_truth_ret['type'] == 'graph':
        print("Two graph are not identical.")
    else:
        print("ground truth: ", ground_truth_ret['data'])
        print("model output: ", ret['data'])

    # Save requestData, code, ground_truth_ret['data'] into a JsonLine file
    with jsonlines.open(OUTPUT_JSONL_PATH, mode='a') as writer:
        writer.write(requestData['original-prompt'])
        writer.write({"Token count input": llm_input_token_count})
        writer.write({"Token count output": llm_output_token_count})
        writer.write({"Result": "Fail"})
        if ground_truth_ret['type'] != 'graph':
            writer.write({"Ground truth exec": ground_truth_ret['data']})
            writer.write({"LLM code exec": ret['data']})
    return None

def ground_truth_check_accu(count, requestData, ground_truth_ret, ret, llm_output_token_count):
    print("Pass the test!")
    count += 1
    # Save requestData, code, ground_truth_ret['data'] into a JsonLine file
    with jsonlines.open(OUTPUT_JSONL_PATH, mode='a') as writer:
        writer.write(requestData['original-prompt'])
        writer.write({"Token count input": llm_input_token_count})
        writer.write({"Token count output": llm_output_token_count})
        writer.write({"Result": "Pass"})
        if ground_truth_ret['type'] != 'graph':
            writer.write({"Ground truth exec": ground_truth_ret['data']})
            writer.write({"LLM code exec": ret['data']})
    return count

def main():
    # create 'output.jsonl' file if it does not exist
    if not os.path.exists(OUTPUT_JSONL_PATH):
        with open(OUTPUT_JSONL_PATH, 'w') as f:
            pass

    graph_json = getGraphData()

    # TODO: replace here by auto-loading from a Json data
    prompt_list = [
        # 8 easy ones
        "How many nodes are in the graph? Return only the number.",
        "How many nodes and edges are in the graph? Return a list.",
        "Add a label app:prod to nodes with address prefix 15.76 and add the label app:test to nodes with address prefix 149.196. Return the networkx graph object.",
        "Show me the unique labels and the number of nodes per label. Return a table with header 'Label', 'Number of Nodes' on the first row.",
        "Remove the label 'type=VM' from all the nodes. Return the networkx graph object.",
        "What are max degree and min degree in the graph? Return a table with with header 'Max degree', 'Min degree' on the first row.",
        "Color all of the nodes with label 'app:prod' purple. Return the networkx graph object.",
        "Color the node with max degree red and min degree green. Return the networkx graph object.",

        # 8 medium ones
        "How many nodes are there that have an edge to nodes with labels app:prod or app:test and doesn't have either of those labels? Return only the number.",
        "Assign a unique color for each /16 IP address prefix and color the nodes accordingly. Return the networkx graph object.",
        "Color the size of the node with max degree green and double it's size. Return the networkx graph object.",
        "Find nodes with top 10 number of degrees, list nodes, labels and number of degrees. Return a table without headers.",
        "Color the nodes that can be connect to nodes with labels app:prod with green. Return the networkx graph object.",
        "Cut the graph into two parts such that the number of edges between the cuts is same. Color two parts with red and blue. Return the networkx graph object.",
        "Identify the unique labels in the graph. Create a new graph with a node for each unique label. For each edge in the old graph, identify the labels of the nodes on either side of the edge. Connect the nodes with those labels in the new graph if they are not already connected by an edge. Return the networkx graph object.",
        "Calculate the total byte weight of edges incident on each node, use kmeans clustering to cluster the total byte weights into 5 clusters, apply the cluster labels as strings to the nodes and pick and assign colors to the nodes based on their cluster labels. Shape the data correctly using numpy before passing it to kmeans. Return the networkx graph object.",

        # 8 hard ones
        "How many maximal cliques are in the graph? Return only the number.",
        "Color the nodes to reflect a heatmap based on the total byte weight of the edges. Return the networkx graph object.",
        "Bisect the network such that the number of nodes on either side of the cut is equal. Color the graph based on the bisection. Return the networkx graph object.",
        "Calculate the total byte weight of edges incident on each node, use kmeans clustering to cluster the total byte weights into 5 clusters. Return the networkx graph object.",
        "How many unique nodes have edges to nodes with label app:prod and doesn't contain the label app:prod? Return only the number.",
        "Show me the unique IP address prefix and the number of nodes per prefix. Return a table without headers.",
        "Delete all edges whose byte weight is less than the median byte weight in the whole graph without using the statistics library. Make sure to compute the median and not the mean. Return the networkx graph object.",
        "What is the average byte weight and connection weight of edges incident on nodes with labels app:prod? Return a table with header 'Average byte weight', 'Average connection weight' on the first row.",

    ]
    userQuery(prompt_list, graph_json)


if __name__=="__main__":
    main()
