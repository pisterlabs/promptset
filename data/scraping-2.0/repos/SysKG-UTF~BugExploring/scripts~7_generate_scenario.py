import ast
import json
from pathlib import Path

import openai
from tqdm import tqdm

from bug_improving.pipelines.generator import ScenarioLinker, ScenarioCombiner
from bug_improving.utils.file_util import FileUtil
from bug_improving.utils.graph_util import GraphUtil
from bug_improving.utils.llm_util import LLMUtil
from bug_improving.utils.path_util import PathUtil
from config import DATA_DIR


def get_bug_id_pairs(seed_bug_id, bugs):
    bug_id_pairs = []
    for bug in bugs:
        bug_id_pairs.append((seed_bug_id, bug.id))
    return bug_id_pairs


if __name__ == "__main__":
    openai.api_key = LLMUtil.OPENAI_API_KEY

    bugs = FileUtil.load_pickle(PathUtil.get_filtered_bugs_filepath())
    # *** seletction ***

    model_name = LLMUtil.GPT4_MODEL_NAME
    # model_name = LLMUtil.TURBO_MODEL_NAME

    # *** seletction ***
    with_instances = bugs
    # with_instances = None

    with_step_cluster = True

    # seed_bug_id = 1742384
    # seed_bug_id = 1757156
    # seed_bug_id = 1678633
    # seed_bug_id = 1764787
    # seed_bug_id = 1708424
    # seed_bug_id = 1741359
    seed_bug_id = 1267592

    foldername = 'scenarios'
    filename = f"{seed_bug_id}"

    GraphUtil.BUGS = bugs
    GraphUtil.get_bug_id_bug_dict(bugs)
    GraphUtil.get_index_cluster_dict(bugs)
    GraphUtil.get_index_cluster_expected_actual_result_dict()

    bug_list, bug_ranking_details_dict = GraphUtil. \
        find_relevant_ranked_bugs_by_bug_id_with_step_type(bugs, seed_bug_id)

    # bug_id_pairs = [(1575516, 1576189)]
    bug_id_pairs = get_bug_id_pairs(seed_bug_id, bug_list[0:10])

    answers = []
    for bug_id_pair in tqdm(bug_id_pairs, ascii=True):
        bug_pair = (bugs.get_bug_by_id(bug_id_pair[0]), bugs.get_bug_by_id(bug_id_pair[1]))
        print(bug_pair[0])
        print(bug_pair[1])
        answer, _ = ScenarioLinker.link_scenario(bug_pair, with_instances, model_name, temperature=0.35)
        answer = ast.literal_eval(answer)
        # print(type(answer))
        answers.append({
            "bug_id_pair": bug_id_pair,
            "answer": answer
        })

        FileUtil.dump_json(Path(DATA_DIR, foldername, f"{filename}.json"), answers)

        answer, _ = ScenarioCombiner.combine_scenario(bug_pair, with_instances, with_step_cluster, model_name)

        answer = ast.literal_eval(answer)
        # print(type(answer))
        answers.append({
            "bug_id_pair": bug_id_pair,
            "answer": answer
        })

        FileUtil.dump_json(Path(DATA_DIR, foldername, f"{filename}.json"), answers)

