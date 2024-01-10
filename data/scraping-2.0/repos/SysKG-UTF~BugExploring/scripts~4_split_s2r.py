import json
from datetime import datetime
from pathlib import Path

import openai
from tqdm import tqdm

from bug_improving.pipelines.constructor import StepSplitter
from bug_improving.utils.file_util import FileUtil
from bug_improving.utils.llm_util import LLMUtil
from bug_improving.utils.path_util import PathUtil
from config import DATA_DIR

if __name__ == "__main__":
    openai.api_key = LLMUtil.OPENAI_API_KEY
    bugs = FileUtil.load_pickle(PathUtil.get_filtered_bugs_filepath())
    step_result_filepath = Path(DATA_DIR, "step")

    with_instances = bugs
    # with_instances = None

    with_step_type = True
    # with_step_type = False

    bug_id_answer_pairs = []
    for index, bug in tqdm(enumerate(bugs), ascii=True):
        print(index)
        print(bug)
        answer = None
        try:
            if bug.description.steps_to_reproduce:
                answer, _ = StepSplitter.split_s2r(bug, with_instances, with_step_type)
            # input()
                ans_json = json.loads(answer)
                bug_id_answer_pairs.append({"bug_id": bug.id, "ans": ans_json})
            else:
                bug_id_answer_pairs.append({"bug_id": bug.id, "ans": []})
        except Exception as e:
            print(f"{e}")
            print(answer)
            pass
        print("************************************************")
        if index % 100 == 0:
            current_datetime = datetime.now()
            FileUtil.dump_json(Path(step_result_filepath, f"bug_id_ans_pairs_{current_datetime}.json"),
                               bug_id_answer_pairs)
            bug_id_answer_pairs = []

    if bug_id_answer_pairs:
        current_datetime = datetime.now()
        FileUtil.dump_json(Path(step_result_filepath, f"bug_id_ans_pairs_{current_datetime}.json"),
                           bug_id_answer_pairs)

