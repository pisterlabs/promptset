import json
import logging
import os
import sys
from dotenv import load_dotenv
from ie_client import IEClient
from openai_client import OpenAIClient
from summarizer import Summarizer


def parse_repo(repo_path, output_path) -> int:
    return os.system(
        f"java -jar ./java-repo-parser.jar -r={repo_path} -o={output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_eval_sum.py <start_idx> <end_idx>")
        exit(1)
    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])

    load_dotenv()

    repo_root_path = "./eval_data/repo"
    repo_list_file_path = "./eval_data/filtered/repo_final.json"
    result_root_path = "./eval_data/sum_result"

    if not os.path.exists(result_root_path):
        os.mkdir(result_root_path)

    logging.basicConfig(level=logging.INFO,
                        format='%(name)s - %(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')
    pipeline_logger = logging.getLogger("pipeline")

    # create clients
    try:
        ie_client = IEClient()
        openie_client = OpenAIClient()

        if not ie_client.check_health():
            raise Exception("Inference Endpoints is not available.")
    except Exception as e:
        pipeline_logger.error(e)
        exit(1)

    with open(repo_list_file_path, "r") as f_repo_list:
        repo_objs = json.load(f_repo_list)

        for idx, repo_obj in enumerate(repo_objs[start_idx:end_idx]):
            try:
                repo_name = repo_obj['repo'].split('/')[-1]
                repo_path = os.path.join(
                    repo_root_path, f"{repo_name}-{repo_obj['sha']}")

                result_dir_path = os.path.join(result_root_path, repo_name)

                if not os.path.exists(result_dir_path):
                    os.mkdir(result_dir_path)

                parse_out_path = os.path.join(
                    result_dir_path, f"parse_out_{repo_name}.json")
                sum_log_path = os.path.join(
                    result_dir_path, f"sum_log_{repo_name}.txt")
                sum_out_path = os.path.join(
                    result_dir_path, f"sum_out_{repo_name}.json")

                # if repo was already summarized, skip it
                if os.path.exists(sum_out_path):
                    pipeline_logger.info(
                        f"{idx + start_idx}th repo: {repo_name} has been summarized.")
                    continue

                # create logger
                sum_logger = logging.getLogger(sum_log_path)
                sum_logger.addHandler(
                    logging.FileHandler(sum_log_path, "w", "utf-8")
                )
                sum_logger.propagate = False  # prevent printing to console

                pipeline_logger.info(
                    f"Summarizing {idx + start_idx}th repo: {repo_name}...")

                # check if existence of path
                if not os.path.exists(repo_path):
                    raise Exception(f"Repo's path does not exist.")

                # parse entire repo using java-repo-parser tool
                if (0 != parse_repo(repo_path, parse_out_path)):
                    raise Exception("Failed to parse repo.")

                # build summary tree for entire repo
                summarizer = Summarizer(sum_logger, ie_client, openie_client)
                with open(parse_out_path, "r") as f_parse_out:
                    repo_obj = json.loads(f_parse_out.read())
                    result = summarizer.summarize_repo(repo_obj)

                    # write result to file
                    with open(sum_out_path, "w") as f_sum_out:
                        f_sum_out.write(json.dumps(result))

                pipeline_logger.info(
                    f"Finished summarizing {idx + start_idx}th repo: {repo_name}")

            except Exception as e:
                pipeline_logger.error(e)
                pipeline_logger.warning(f'Stop at {idx + start_idx}')
                break

    logging.shutdown()
