import sys, os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain.callbacks import get_openai_callback
from tqdm import tqdm

sys.path.append(os.environ.get("PROJECT_PATH"))
os.chdir(os.environ.get("PROJECT_PATH"))

import utils.vector_store as vs
import utils.db as db


def main():
    vs.validate_openai_env()

    arxiv_codes = db.get_arxiv_id_list(db.db_params, "summary_notes")
    title_map = db.get_arxiv_title_dict(db.db_params)
    done_codes = db.get_arxiv_id_list(db.db_params, "recursive_summaries")
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[::-1]

    with get_openai_callback() as cb:
        for arxiv_code in tqdm(arxiv_codes):
            paper_notes = db.get_extended_notes(arxiv_code)
            paper_title = title_map[arxiv_code]

            ## Insert copywriter's summary into the database.
            narrative = vs.convert_notes_to_narrative(paper_title, paper_notes)
            copywritten = vs.copywrite_summary(paper_title, narrative)
            db.insert_recursive_summary(arxiv_code, copywritten)

        print(cb)

    print("Done!")


if __name__ == "__main__":
    main()
