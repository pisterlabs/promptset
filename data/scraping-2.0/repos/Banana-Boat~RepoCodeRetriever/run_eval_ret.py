import json
import logging
import os
import sys
from dotenv import load_dotenv
from tqdm import tqdm

from openai_client import OpenAIClient
from retriever import Retriever
from text_sim_calculator import TextSimCalculator
from sim_retriever import SimRetriever


if __name__ == "__main__":
    start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    load_dotenv()

    data_file_path = "./eval_data/filtered/data_final.jsonl"
    sum_result_root_path = "./eval_data/sum_result"
    ret_log_dir_path = "./eval_data/ret_log"
    ret_result_file_path = "./eval_data/ret_result.jsonl"

    if not os.path.exists(ret_log_dir_path):
        os.mkdir(ret_log_dir_path)

    logging.basicConfig(level=logging.INFO,
                        format='%(name)s - %(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')
    pipeline_logger = logging.getLogger("pipeline")

    # create client for OpenAI
    try:
        openai_client = OpenAIClient()
    except Exception as e:
        pipeline_logger.error(e)
        exit(1)

    # create similarity caculator
    text_sim_calculator = TextSimCalculator()
    # create retriever
    retriever = Retriever(
        openai_client, text_sim_calculator)

    # create sim_retriever(ablation experiment)
    # retriever = SimRetriever(text_sim_calculator)

    with open(data_file_path, "r") as f_data, open(ret_result_file_path, "a") as f_ret_result:
        data_objs = [json.loads(line) for line in f_data.readlines()]

        for idx, data_obj in enumerate(tqdm(data_objs[start_idx:])):
            try:
                query = data_obj['query']
                repo_name = data_obj['repo'].split('/')[-1]

                sum_out_path = os.path.join(
                    sum_result_root_path, repo_name, f"sum_out_{repo_name}.json")
                if not os.path.exists(sum_out_path):
                    raise Exception("Summary output path does not exist.")

                # create logger
                ret_log_path = os.path.join(
                    ret_log_dir_path, f"ret_log_{data_obj['id']}.txt")
                ret_logger = logging.getLogger(ret_log_path)
                ret_logger.addHandler(
                    logging.FileHandler(ret_log_path, "w", "utf-8")
                )
                ret_logger.propagate = False  # prevent printing to console

                with open(sum_out_path, "r") as f_sum_out:
                    repo_sum_obj = json.load(f_sum_out)

                    # retrieve the result
                    is_error, res_obj = retriever.retrieve(
                        query, repo_sum_obj, ret_logger)
                    if is_error:
                        raise Exception("An error occurred during retrieval.")

                    # write result to file
                    obj = {
                        'id': data_obj['id'],
                        'is_found': res_obj['is_found'],
                        'is_query_expanded': res_obj['is_query_expanded'],
                        'path': res_obj['path'],
                        'ret_times': res_obj['ret_times'],
                    }
                    f_ret_result.write(json.dumps(obj) + '\n')
                    f_ret_result.flush()

            except Exception as e:
                pipeline_logger.error(e)
                pipeline_logger.warning(f'Stop at {idx + start_idx}')
                break

    logging.shutdown()
