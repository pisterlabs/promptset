import argparse
import networkx as nx
import pickle as pkl
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from threading import Thread
from time import sleep
import os
from utils import task_handler, answer_cleasing, evaluate
from functools import partial
import time
from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate

import wandb

os.environ["OPENAI_API_TYPE"]    = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-3-15-preview"
os.environ["OPENAI_API_BASE"]    = "XXX" #
os.environ["OPENAI_API_KEY"]     = "XXX"

def prompting(task, input, question_head, instructer, gramma, example, tail, graph,  change_order, use_gramma):
    
    if task == "size":
        question = question_head
    elif task == "degree":
        question = question_head + str(input) + " is ?\n"
    elif task == "hasedge":
        question = "Is there an edge between " + str(input[0]) + " and " + str(input[1]) + " in this graph?\n"
    elif task == "hasattr":
        question = question_head + str(input) + " is ?\n"
    elif task == "diameter":
        question = question_head
    elif task == "clustering":
        question = question_head + str(input) + " is? \n"
    
    if not use_gramma:
        gramma = ""
    
    if change_order:
        prompt = instructer + gramma + example + question + " "  + graph + tail
    else:
        prompt = graph + instructer + gramma + example + question + " "  + tail
        
    print(prompt)
        
    return prompt
     

def main(config, seed=0):
    
    GPT = AzureOpenAI(
    deployment_name="text-davinci-003",
    model_name="text-davinci-003",
    temperature=0.3)
        
    if config.use_role_prompting:
        instructer = "You are a brilliant graph analyzer and knows every thing about graph. You know every thing about the structure of graphs and how to compute graph sturcture metrics. \n"
    else:
        instructer = ""
    
    if config.format == "GML":
        prefix  = "./input_arxiv/GML"
        reader  = nx.read_gml
        postfix = ".gml"
        gramma  = "<GML grammar> Each node has a unique id and a label. Node is labeled with node [ id ... label ... ... ], edge is labeled with edge [ source ... target ... ... ] \n"
    elif config.format == "GraphML":
        prefix  = "./input_arxiv/GraphML"
        reader  = partial(nx.read_graphml)
        postfix = ".graphml"
        gramma  = "<GraphML grammar> Each node has a unique id and a label. Node is labeled with <node id=...> <data key=...> ... </data> </node> and edge is labeled with <edge source= ... target=... > <data key=...>...</data> </edge> \n"
    elif config.format == "EdgeList":
        prefix  = "./input_arxiv/EdgeList"
        reader  = partial(nx.read_edgelist, delimiter="\t")
        postfix = ".edgelist"
        gramma  = "<Edge list grammar> Each row is an edge with the format of node1, node2, {...}. \n"
    elif config.format == "AdjList":
        prefix  = "./input_arxiv/AdjList"
        reader  = partial(nx.read_adjlist,  delimiter="\t")
        postfix = ".adjlist"
        gramma  = "<Adjacency list grammar> Each row is a node (at the first) with its connected neighbors (followed by the first node) with the format of node1, node2, node3, ... \n"
    
    example = ""
    tail    = ""
    if "one_shot" in config.method:
        if config.task == "degree":
            example = "For example, a node has degree of x if there are x edges (or other nodes) connecting to it. \n"
        elif config.task == "hasedge":
            example = "For example, node A and node B are connected if there is an edge between them in this graph. However, node A and node B is not connected if there is no edge between them in this graph. \n"
        elif config.task == "hasattr":
            example = "For example, node A has title Efficient Algorithm for Recommendation if its title attribute value is Efficient Algorithm for Recommendation. \n"
        elif config.task == "size":
            if config.format != "GraphML":
                random = np.random.randint(0, 100)
                example_graph = os.path.join(prefix, "graph_"+str(random)+postfix)
                example_graph_nx = reader(example_graph)
                with open(example_graph, "r") as fp:
                    example_graph = fp.read()
                example = "For example, the following graph " + example_graph + " has " + str(example_graph_nx.number_of_nodes()) + " nodes and " + str(example_graph_nx.number_of_edges()) + " edges. \n"
            else:
                example = "The size of a graph is the number of nodes and the number of edges. \n"
        elif config.task == "clustering":
            example = "For example, the clustering coefficient of a node is 0.33 if the node has 3 neighbors and only 2 of the 3 neighbors are connected. \n"
        elif config.task == "diameter":
            example = "For example, the diameter of a graph is 3 if the maximum distance between any node pairs in this graph is 3. \n"
    if "cot" in config.method:
        tail = "Let's think step by step. \n"

    # print(example)
    accs = []
    for _ in tqdm(range(1)):
        for idx, seed in tqdm(enumerate(range(100)), total=100):
            graph_file = os.path.join(prefix, "graph_"+str(seed)+postfix)
            with open(graph_file, "r") as fp:
                graph = fp.read()
            graph_nx = reader(graph_file)
                
            question_head, true_answer = task_handler(config.task, graph_nx, config)
            # print(question, answer)
            # print(question_head, true_answer)
            if config.task == "size":
                if "one_shot" in config.method and config.format != "GraphML":
                    random = np.random.randint(0, 100)
                    example_graph = os.path.join(prefix, "graph_"+str(random)+postfix)
                    example_graph_nx = reader(example_graph)
                    with open(example_graph, "r") as fp:
                        example_graph = fp.read()
                    example = "For example, the size of the following graph " + example_graph + " is " + str(example_graph_nx.number_of_nodes()) + " nodes and " + str(example_graph_nx.number_of_edges()) + " edges. \n"
                elif config.format == "GraphML":
                    example = "The size of a graph is the number of nodes and the number of edges. \n"
                else:
                    example = ""
                prompt = prompting(config.task, None, question_head, instructer, gramma, example, tail, graph, change_order=config.change_order, use_gramma=config.use_format_explain)
                answer   = GPT(prompt)
                if "cot" in config.method:
                    answer = GPT(prompt + " " + answer + " Therefore the answer is " )
                pred       = answer_cleasing(config, None, answer)
                print(pred, true_answer)
                accs.append(evaluate(pred, true_answer))
                
            elif config.task == "degree":
                
                predictions = []
                
                for node in graph_nx:
                    
                    prompt = prompting(config.task, node, question_head, instructer, gramma, example, tail, graph, change_order=config.change_order, use_gramma=config.use_format_explain)
                    answer     = GPT(prompt)
                    if "cot" in config.method:
                        answer = GPT(prompt + " " + answer + " Therefore the answer is " )
                    pred       = answer_cleasing(config, str(node)[1:], answer)
                    
                    predictions.append(pred)
                
                print(predictions, true_answer)
                acc = evaluate(predictions, true_answer)
                accs.append(acc)
                wandb.log({"epoch_acc": acc})
                
            elif config.task == "hasedge":
                
                predictions = []
                for edge in list(graph_nx.edges())[:10]:
                    prompt   = prompting(config.task, edge, question_head, instructer, gramma, example, tail, graph, change_order=config.change_order, use_gramma=config.use_format_explain)
                    answer   = GPT(prompt)
                    if "cot" in config.method:
                        answer = GPT(prompt + " " + answer + " Therefore the answer is " )
                    pred     = answer_cleasing(config, None, answer)
                    predictions.append(pred)
                graph_nx_com = nx.complement(graph_nx)
                for edge in list(graph_nx_com.edges())[:10]:
                    prompt   = prompting(config.task, edge, question_head, instructer, gramma, example, tail, graph, change_order=config.change_order, use_gramma=config.use_format_explain)
                    answer   = GPT(prompt)
                    if "cot" in config.method:
                        answer = GPT(prompt + " " + answer + " Therefore the answer is " )
                    pred       = answer_cleasing(config, None, answer)
                    predictions.append(pred)
                
                acc = evaluate(predictions, true_answer)
                accs.append(acc)
                wandb.log({"epoch_acc": acc})
                
            elif config.task == "hasattr":
                
                predictions = []
                for idx, node in enumerate(graph_nx):
                    prompt = prompting(config.task, node, question_head, instructer, gramma, example, tail, graph, change_order=config.change_order, use_gramma=config.use_format_explain)
                    answer   = GPT(prompt)
                    if "cot" in config.method:
                        answer = GPT(prompt + " " + answer + " Therefore the answer is " )
                    pred       = answer_cleasing(config, None, answer.lower(), groud_truth=true_answer[idx].lower())
                    predictions.append(pred)
                acc = evaluate(predictions, true_answer, match=True)
                print(acc)
                accs.append(acc)
                wandb.log({"epoch_acc": acc})
                
            elif config.task == "clustering":
                
                predictions = []
                # print(graph_nx)
                for node in graph_nx:
                    prompt = prompting(config.task, node, question_head, instructer, gramma, example, tail, graph, change_order=config.change_order, use_gramma=config.use_format_explain)
                    answer   = GPT(prompt)
                    
                    if "cot" in config.method:
                        # print(answer)
                        answer = GPT(prompt + " " + answer + " Therefore the answer is " )
                        # print(answer)
                    # print(prompt + " " + answer + " Therefore the answer is " )
                    pred       = answer_cleasing(config, None, answer)
                    predictions.append(pred)
                    # time.sleep(5)
                print(predictions, true_answer)
                acc = evaluate(predictions, true_answer)
                accs.append(acc)
                wandb.log({"epoch_acc": acc})
                
            elif config.task == "diameter":
                
                prompt = prompting(config.task, None, question_head, instructer, gramma, example, tail, graph, change_order=config.change_order, use_gramma=config.use_format_explain)
                answer   = GPT(prompt)
                if "cot" in config.method:
                    answer = GPT(prompt + " " + answer + " Therefore the answer is " )
                pred       = answer_cleasing(config, None, answer)
                predictions = [pred]
                acc = evaluate(predictions, true_answer)
                accs.append(acc)
                wandb.log({"epoch_acc": acc})
                    
    accs = np.array(accs)
    print(np.mean(accs), np.std(accs))
    wandb.log({"acc": np.mean(accs), "std": np.std(accs)})
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--format",             type=str, default="EdgeList",   help="Input format to use. ")
    parser.add_argument("--dataset",            type=str, default="obgn-arxiv", help="The dataset to use. ")
    parser.add_argument("--method",             type=str, default="one_shot",  help="The method to use. ")
    parser.add_argument("--use_format_explain", type=int, default=1,            help="whether use change order. ")
    parser.add_argument("--use_role_prompting", type=int, default=1,            help="whether use change order. ")
    parser.add_argument("--change_order",       type=int, default=1,            help="whether use change order. ")
    parser.add_argument("--self_augument",      type=int, default=0,            help="whether use self-aug. ")
    parser.add_argument("--task",               type=str, default="clustering",     help="The task to conduct. ")
    args = parser.parse_args()
    
    wandb.init(project="GraphBench", config=args)
    
    main(args)
    
    # print("zero_shot: ")
    # main(args)
    # args.change_order = True
    # print("zero_shot + change_order: ")
    # main(args)
    # args.method = "one_shot"
    # args.change_order = False
    # print("one_shot:")
    # main(args)
    # args.method = "one_shot"
    # args.change_order = True
    # print("one_shot + change_order: ")
    # main(args)
    
