import swissnyf
from swissnyf.pipeline import *
from swissnyf.retriever import *
from swissnyf.utils import *
from swissnyf.configs import *
#from swissnyf.tools_util import *
import argparse
import os
import sys
import copy
import pickle
import json
import re
from llama_index.embeddings import OpenAIEmbedding
from llama_index.agent import ReActAgent
import re
from tqdm import tqdm
from llama_index.tools.function_tool import FunctionTool
from typing import List
from llama_index.agent.react.formatter import  get_react_tool_descriptions
from llama_index.llms.base import LLM
from functools import wraps
from collections.abc import Iterable
import inspect, itertools 
from tqdm import tqdm
from typing import List

from functools import wraps
from collections.abc import Iterable
from abc import abstractclassmethod
from typing import Optional, Dict, List, Tuple
from sentence_transformers import SentenceTransformer, util
from llama_index.llms import AzureOpenAI, OpenAI
from llama_index.embeddings import OpenAIEmbedding
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

import time


from dotenv import load_dotenv, find_dotenv
dirname = os.path.join(currentdir, '.env')
load_dotenv(dirname)

def main():
    parser = argparse.ArgumentParser()

    # add args.config to parser
    parser.add_argument("--config", type=str, default=f"{currentdir}/configs/config.yaml", help="config file")

    # add args.tools to parser
    parser.add_argument("--tools", type=str, default=f"{currentdir}/data/tools.yaml", help="tools file")

    # add args.pipeline to parser
    parser.add_argument("--pipeline", type=str, default="topgun", help="pipeline file")

    # add args.retriever to parser
    parser.add_argument("--retriever", type=str, default="instructxl", help="retriever file")

    # add optional args.input to parser where query is a jsonl file of queries
    parser.add_argument("--input", type=str, default=f"{currentdir}/data/input.txt", help="input file", required=False)

    # add args.output to parser
    parser.add_argument("--output", type=str, default="./output.jsonl", help="output file", )

    # add args.query to parser where query is a string
    parser.add_argument("--query", type=str, help="query string", required=False)

    # add arg.llm to parser
    parser.add_argument("--llm", type=str, default="openai", help="llm")

    # add args.verbose to parser
    parser.add_argument("--verbose", type=bool, default=False, help="verbose")

    args = parser.parse_args()


    # use default args if no args are passed
    

    if args.config is not None:
        config = Config.parse_yaml_file(args.config)
    
    if args.tools is not None:
        tools = Tools(args.tools)
    
    verbose = args.verbose
    
    all_tools_list = tools.get_tools_list()
    all_tools_names = tools.get_tool_names()
    all_tools_desc = tools.get_tools__desc_str()

    # print("All tools are loaded and set up")

    #### need to do proper logging after each level of pipeline
    # instantiate logger object

    # instantiate retriever object based on if condition from args
    if args.retriever is not None:
        if args.retriever == "gear":
            # need to populate and instantiate them properly
            retriever = GearRet(top_k = 9, verbose = False)
        elif args.retriever == "toolbench":
            retriever = ToolBenchRet(top_k = 9, verbose = False)
        elif args.retriever == "instructxl":
            retriever = InstructRet(top_k = 9, verbose = False)
        else:
            raise Exception("retriever method not supported")
    
    # print("Retriever is initalised")

    retriever.set_tool_def(all_tools_list)

    # print("Retriever is set up with all tools")

    #### need to work on this populat with keys and params from config
    # instantiate llm object based on if condition from args
    if args.llm is not None:
        if args.llm == "azure":
            model = os.environ["OPENAI_API_MODEL"]
            deployment = os.environ["OPENAI_API_DEPLOYMENT"]
            llm = AzureOpenAI(deployment_id=deployment, model=model, engine=deployment, temperature=0.01)
        elif args.llm == "openai":
            model = os.environ["OPENAI_API_MODEL"]
            llm = OpenAI(model=model, temperature=0.01)
        else:
            raise Exception("llm not supported")

    # print("LLM is initalised")

    # # instantiate reversechain and topgun pipeline object as per pipeline file from args param pipeline
    if args.pipeline is not None:
        if args.pipeline == "reversechain":
            pipeline = ReverseChain(filter_method=retriever, llm=llm)
        elif args.pipeline == "topgun":
            pipeline = TopGun(filter_method=retriever, llm=llm)
        else:
            raise Exception("pipeline method not supported")
    
    # print("Pipeline is initalised")

    # set tools to pipeline
    pipeline.set_tools(all_tools_desc, all_tools_names)

    # print("Pipeline is set up with all tools")


    # demo of pipeline on a query
    if args.query is not None:
        query = args.query
        max_retries = 5
        completed = False
        retries = 0
        while not completed and retries < max_retries:
            try:
                response = pipeline.query(query) 
                completed = True
                # write output to output file
                with open(args.output, "a") as f:
                    f.write(response)
            except Exception as e:
                print(f"Retrying query: {query},  {e}")
                retries+=1
       
    elif args.input is not None:
        # read input file
        with open(args.input, "r") as f1:
            queries = f1.readlines()
        # run pipeline on each query
        for query in queries:
            max_retries = 3
            completed = False
            retries = 0
            while not completed and retries < max_retries:
                try:
                    print("Query:", query)
                    response = pipeline.query(query) 
                    completed = True
                    # write output to output file
                    with open(args.output, "a") as f2:
                        f2.write(response)
                except Exception as e:
                    print(f"Retrying query: {query},  {e}")
                    retries+=1
    
    # print("Pipeline is run on query")
   

if __name__ == "__main__":
    main()   












