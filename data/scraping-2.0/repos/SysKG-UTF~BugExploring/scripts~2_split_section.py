import json
from datetime import datetime
from pathlib import Path

import openai
from tqdm import tqdm

from bug_improving.pipelines.constructor import SecSplitter
from bug_improving.utils.file_util import FileUtil
from bug_improving.utils.llm_util import LLMUtil
from bug_improving.utils.path_util import PathUtil
from config import DATA_DIR

if __name__ == "__main__":
    openai.api_key = LLMUtil.OPENAI_API_KEY
    bugs = FileUtil.load_pickle(PathUtil.get_filtered_bugs_filepath())
    result_filepath = Path(DATA_DIR, "section")
    # with_instances = None
    with_instances = bugs

    bug_id_answer_pairs = []
    for index, bug in tqdm(enumerate(bugs), ascii=True):
        print(index)
        print(bug)
        answer = None
        try:
            answer, _ = SecSplitter.split_section(bug, with_instances)
            ans_json = json.loads(answer)
            bug_id_answer_pairs.append({"bug_id": bug.id, "ans": ans_json})
            # bug.description.get_sections_from_dict(answer)
        except Exception as e:
            print(f"{e}")
            print(answer)
            pass
        print("************************************************")
        if index % 100 == 0:
            current_datetime = datetime.now()
            FileUtil.dump_json(Path(result_filepath, f"bug_id_ans_pairs_{current_datetime}.json"),
                               bug_id_answer_pairs)
            bug_id_answer_pairs = []

    if bug_id_answer_pairs:
        current_datetime = datetime.now()
        FileUtil.dump_json(Path(result_filepath, f"bug_id_ans_pairs_{current_datetime}.json"),
                           bug_id_answer_pairs)
