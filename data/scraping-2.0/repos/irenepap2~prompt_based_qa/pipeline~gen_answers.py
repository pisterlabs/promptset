import os
import json
import argparse
import pandas as pd
import pipeline.utils as utils

from pipeline.constants import COLLECTION_PATH
from pipeline.get_collection import get_queries_from_file
from tqdm import tqdm


def get_collection_from_json(user_query, retrieval_method):

    with open(os.path.join(COLLECTION_PATH, retrieval_method, user_query + ".json"), "r", encoding="utf-8") as f:
        collection = json.load(f)
    return collection


def get_answer_per_chunk_per_paper(collection, prompt_filename, model):

    extract_q = collection["extract_query"]
    hits = collection["hits"]
    answers_dict = {}
    for hit in tqdm(hits[:10]):
        chunk_responses = []
        for passage in hit["passages"]:
            if prompt_filename == "zeta_alpha_prompt":        
                prompt = utils.construct_prompt(
                    filename=prompt_filename + ".txt", 
                    prompt_params_dict={
                        "extract_q" : extract_q,
                        # "doc_title" : hit["metadata"]["title"],
                        "doc_excerpt" : passage,
                    },
            )
            response = utils.get_model_response(prompt, model)["choices"][0]["text"]
            chunk_responses.append(response)
        guid = hit["guid"]
        answers_dict[guid] = chunk_responses

    return answers_dict


def get_answer_per_paper(collection, prompt_filename, model):

    extract_q = collection["extract_query"]
    hits = collection["hits"]
    answers_dict = {}
    for hit in tqdm(hits):
        prompt = utils.construct_prompt(
                filename=prompt_filename + ".txt", 
                prompt_params_dict={
                    "question" : extract_q,
                    "context" : hit["metadata"]["abstract"],
                },
        )
        response = utils.get_model_response(prompt, model)["choices"][0]["text"]
        guid = hit["guid"]
        answers_dict[guid] = {
            "answer": response,
            "title": hit["metadata"]["title"],
            "evidence": hit["metadata"]["abstract"],
        }
    return answers_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # either user_query or query_file is required
    group = parser.add_mutually_exclusive_group(required=True)
    # read query from command line
    group.add_argument("--query", type=str, help="Query to search for documents")
    # read query from file
    group.add_argument("--query_file", type=argparse.FileType("r"), help="File containing queries to search for documents")
    # choose prompt to use
    parser.add_argument("--prompt_filename", type=str, default="zeta_alpha_prompt", help="Filename of prompt to use")
    # choose between knn or keyword retrieval
    parser.add_argument("--retrieval_method", type=str, default="knn", help="Retrieval method to use (knn or keyword)")
    # choose which model to use
    parser.add_argument("--model", type=str, choices=["text-davinci-003", "code-davinci-002", "gpt-3.5-turbo"], required=True, help="Choose between the OpenAI large language models")
    # choose how many passages to use
    parser.add_argument("--num_passages", type=int, default=3, help="Number of passages retrieved to use in prompt (only with chunk retrieval)")
    # choose whether to use chunk retrieval / Title + Abstract / Random etc.
    parser.add_argument("--chunk_retrieval", default=True, help="Use chunk retrieval")

    args = parser.parse_args()

    if args.query_file:
        all_queries = get_queries_from_file(args.query_file)
        for query_id, query_types in all_queries.items():
            query = query_types["query"]
            collection = get_collection_from_json(query, args.retrieval_method)
            # check if answer already exists
            # if os.path.exists(os.path.join("data", "collections", args.retrieval_method, args.prompt_filename, f"{query_id}_{query}.csv")):
            #     print(f"Answer for query {query} already exists. Skipping...")
            #     continue
            # check document length

            print(f"Getting answer per document from OpenAI for query: {query}...")
            answers_dict = get_answer_per_paper(collection, args.prompt_filename, args.model)

            # create directory if it doesn't exist
            if not os.path.exists(os.path.join("data", "answers", args.retrieval_method, args.prompt_filename)):
                os.makedirs(os.path.join("data", "answers", args.retrieval_method, args.prompt_filename))
            print(f"Saving answers to data/answers/{args.retrieval_method}/{args.prompt_filename}...")
            with open(os.path.join("data", "answers", args.retrieval_method, args.prompt_filename, f"{query}.json"), "w") as f:
                json.dump(answers_dict, f, indent=4)            
            
            # add column with answer to existing csv file in data/collections
            csv_input = pd.read_csv(os.path.join("data", "collections", args.retrieval_method, "csv_files", f"{query_id}_{query}.csv"))
            csv_input["gen_answer"] = answers_dict.values()
            # create directory if it doesn't exist
            if not os.path.exists(os.path.join("data", "collections", args.retrieval_method, args.prompt_filename)):
                os.makedirs(os.path.join("data", "collections", args.retrieval_method, args.prompt_filename))
            csv_input.to_csv(os.path.join("data", "collections", args.retrieval_method, args.prompt_filename, f"{query_id}_{query}.csv"), index=False)
    
    else:
        print("Getting answer per document from OpenAI...")
        collection = get_collection_from_json(args.query, args.retrieval_method)
        answers_dict = get_answer_per_paper(collection, args.prompt_filename, args.model)

        # create directory if it doesn't exist
        if not os.path.exists(os.path.join("data", "answers", args.retrieval_method, f"{args.prompt_filename}")):
            os.makedirs(os.path.join("data", "answers", args.retrieval_method, f"{args.prompt_filename}"))
        
        # save answers to json file
        print(f"Saving answers to data/answers/{args.retrieval_method}/{args.prompt_filename}...")
        with open(os.path.join("data", "answers", args.retrieval_method, f"{args.prompt_filename}", f"{args.query}.json"), "w") as f:
            json.dump(answers_dict, f, indent=4)
