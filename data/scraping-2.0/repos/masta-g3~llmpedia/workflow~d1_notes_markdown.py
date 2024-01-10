import sys, os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import get_openai_callback
from tqdm import tqdm

sys.path.append(os.environ.get("PROJECT_PATH"))
os.chdir(os.environ.get("PROJECT_PATH"))

import utils.vector_store as vs
import utils.paper_utils as pu
import utils.db as db


def main():
    vs.validate_openai_env()
    title_map = db.get_arxiv_title_dict(db.db_params)
    arxiv_codes = db.get_arxiv_id_list(db.db_params, "summary_notes")
    done_codes = db.get_arxiv_id_list(db.db_params, "summary_markdown")
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[::-1]

    with get_openai_callback() as cb:
        for arxiv_code in tqdm(arxiv_codes):
            paper_notes = db.get_extended_notes(arxiv_code=arxiv_code, expected_tokens=4000)
            paper_title = title_map[arxiv_code]

            ## Convert notes to Markdown format and store.
            markdown_notes = vs.convert_notes_to_markdown(paper_title, paper_notes, model="GPT-4-Turbo")
            markdown_df = pd.DataFrame({"arxiv_code": [arxiv_code], "summary": [markdown_notes], "tstp": [pd.Timestamp.now()]})
            db.upload_df_to_db(markdown_df, "summary_markdown", db.db_params)
            print(cb)

    print("Done!")


if __name__ == "__main__":
    main()
