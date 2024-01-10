import sys, os
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.environ.get("PROJECT_PATH"))
os.chdir(os.environ.get("PROJECT_PATH"))

import pandas as pd
from tqdm import tqdm
import tiktoken

from langchain.callbacks import get_openai_callback
import utils.paper_utils as pu
import utils.vector_store as vs
import utils.db as db

token_encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
LOCAL_PAPER_PATH = os.path.join(os.environ.get("PROJECT_PATH"), "data", "summaries")
RETRIES = 3


def main():
    ## Health check.
    vs.validate_openai_env()

    ## Get paper list.
    arxiv_codes = db.get_arxiv_id_list(pu.db_params, "summary_notes")
    existing_papers = db.get_arxiv_id_list(pu.db_params, "summaries")
    arxiv_codes = list(set(arxiv_codes) - set(existing_papers))

    arxiv_codes = sorted(arxiv_codes)[::-1]
    with get_openai_callback() as cb:
        for arxiv_code in tqdm(arxiv_codes):
            new_content = db.get_extended_notes(arxiv_code, expected_tokens=6000)

            ## Try to run LLM process up to 3 times.
            success = False
            for i in range(RETRIES):
                try:
                    summary = vs.review_llm_paper(new_content)
                    success = True
                    break
                except Exception as e:
                    print(f"\nFailed to run LLM for '{arxiv_code}'. Attempt {i+1}/3.")
                    print(e)
                    continue
            if not success:
                print(f"Failed to run LLM for '{arxiv_code}'. Skipping...")
                continue

            ## Extract and combine results.
            result_dict = summary.json()
            pu.store_local(result_dict, arxiv_code, "summaries")

            ## Store on DB.
            data = pu.convert_innert_dict_strings_to_actual_dicts(result_dict)
            ## ToDo: Legacy, remove.
            if "applied_example" in data["takeaways"]:
                data["takeaways"]["example"] = data["takeaways"]["applied_example"]
                del data["takeaways"]["applied_example"]

            flat_entries = pu.transform_flat_dict(
                pu.flatten_dict(data), pu.summary_col_mapping
            )
            flat_entries["arxiv_code"] = arxiv_code
            flat_entries["tstp"] = pd.Timestamp.now()
            db.upload_to_db(flat_entries, pu.db_params, "summaries")
            # print(f"Added '{arxiv_code}' to summaries table.")

    print("Done!")
    print(cb)


if __name__ == "__main__":
    main()
