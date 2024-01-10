# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from dotenv import load_dotenv
import os
from prototxt_parser.prototxt import parse
from ai_models import pyGraphNetExplorer, llm_input_token_count
import networkx as nx
import jsonlines
import random
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
OUTPUT_JSONL_PATH = 'logs/malt_log.jsonl'

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return cb.total_tokens

def getGraphData():
    input_string = open("../data/malt-example-sample.txt").read()
    parsed_dict = parse(input_string)

    # Load MALT data
    G = nx.DiGraph()

    # Insert all the entities as nodes
    for entity in parsed_dict['entity']:
        # Check if the node exists
        if entity['id']['name'] not in G.nodes:
            G.add_node(entity['id']['name'], type=[entity['id']['kind']], name=entity['id']['name'])
        else:
            G.nodes[entity['id']['name']]['type'].append(entity['id']['kind'])
        # Add all the attributes
        for key, value in entity.items():
            if key == 'id':
                continue
            for k, v in value.items():
                G.nodes[entity['id']['name']][k] = v

    # Insert all the relations as edges
    for relation in parsed_dict['relationship']:
        G.add_edge(relation['a']['name'], relation['z']['name'], type=relation['kind'])

    rawData = json_graph.node_link_data(G)

    return rawData, G

def node_attributes_are_equal(node1_attrs, node2_attrs):
    # Check if both nodes have the exact same set of attributes
    if set(node1_attrs.keys()) != set(node2_attrs.keys()):
        return False

    # Check if all attribute values are equal
    for attr_name, attr_value in node1_attrs.items():
        if attr_value != node2_attrs[attr_name]:
            return False

    return True

def userQuery(prompt_list, graph_json, G):
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
            print("Calling model")
            answer = pyGraphNetExplorer.run(requestData['llm-prompt'])
            llm_output_token_count = count_tokens(pyGraphNetExplorer, answer)
            print("model returned")

            if answer.startswith("Answer:\n'''"):  # For GPT-4
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

    graph_json, G = getGraphData()

    # TODO: replace here by auto-loading from a Json data
    prompt_list = [
        # 3 easy ones
        "List all ports that are contained by packet switch ju1.a1.m1.s2c1. Return a list.",
        "Add a new switch 'ju1.a1.m1.s3c9' on jupiter 1, aggregation block 1, domain 1, with 5 ports. You need to add an edge between new switch and existing topology at each layer. Return the new graph.",
        "Update the physical_capacity_bps from 1000 Mbps to 4000 Mbps on ju1.a1.m1.s2c2.p14. Convert Mbps to bps before the update. Return the new graph.",

        # 3 medium ones
        "What is the bandwidth on ju1.a2.m1.s2c2? Note that first you need to list all port nodes that are contained by packet switch ju1.a2.m1.s2c2. Then sum the attribute physical_capacity_bps on the port nodes together. Output bandwidth unit should be in Mbps. Return only the number.",
        "What is the bandwidth on each AGG_BLOCK? Return a list. Note that AGG_BLOCK contains PACKET_SWITCH, PACKET_SWITCH contains PORT. Then sum the node attribute physical_capacity_bps on the port nodes together. Output bandwidth unit should be in Mbps. Return a table with header 'AGG_BLOCK', 'Bandwidth' on the first row.",
        "Find the first and the second largest Chassis by capacity on 'ju1.a1.m1'. Note that Chassis contains multiple PACKET_SWITCH from different spine block, PACKET_SWITCH contains PORT. Then sum the node attribute physical_capacity_bps on the port nodes together. Output bandwidth unit should be in Mbps. Return a table with header 'Chassis', 'Bandwidth' on the first row.",

        # 3 hard ones
        "Provide a graph that contains all SUPERBLOCK and AGG_BLOCK and plot their hierarchy. Return the new graph.",
        "Remove packet switch 'ju1.a1.m1.s2c4' out from Chassis c4, how to balance the capacity between Chassis? Note that Chassis contains multiple PACKET_SWITCH from different spine block, PACKET_SWITCH contains PORT. Then sum the node attribute physical_capacity_bps on the port nodes together. Return the balanced graph.",
        "Remove five ports from each packet switches ju1.a1.m1.s2c1, ju1.a1.m1.s2c2, ju1.a1.m1.s2c3, ju1.a1.m1.s2c4, ju1.a1.m1.s2c5. Make sure after the removal the capacity between switches is still balanced? Note that PACKET_SWITCH contains PORT. Capacity is the sum of node attribute physical_capacity_bps on the port nodes. Return the list of ports that will be moved."
    ]
    userQuery(prompt_list, graph_json, G)


if __name__=="__main__":
    main()
