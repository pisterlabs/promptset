import sys, os
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.environ.get("PROJECT_PATH"))
os.chdir(os.environ.get("PROJECT_PATH"))

import re, json
import time
from tqdm import tqdm

from langchain.callbacks import get_openai_callback

import utils.paper_utils as pu
import utils.vector_store as vs
import utils.db as db

def update_gist(gist_id, gist_filename, paper_list):
    """Update the gist with the current queue."""
    gist_url = pu.update_gist(
        os.environ["GITHUB_TOKEN"],
        gist_id,
        gist_filename,
        "Updated LLM queue.",
        "\n".join(paper_list),
    )
    return gist_url


def main():
    vs.validate_openai_env()
    parsed_list = []

    ## Get paper list.
    gist_id = "1dd189493c1890df6e04aaea6d049643"
    gist_filename = "llm_queue.txt"
    paper_list = pu.fetch_queue_gist(gist_id, gist_filename)

    ## Check local files.
    local_codes = pu.get_local_arxiv_codes("arxiv_text")
    nonllm_papers = pu.get_local_arxiv_codes("nonllm_arxiv_text")

    ## Remove duplicates.
    paper_list = list(set(paper_list) - set(local_codes) - set(nonllm_papers))
    paper_list_iter = paper_list[:]

    ## Iterate.
    gist_url = None
    with get_openai_callback() as cb:
        for paper_name in tqdm(paper_list_iter):
            time.sleep(10)
            # existing = pu.check_if_exists(
            #     paper_name, existing_paper_names, existing_paper_ids
            # )
            #
            # ## Check if we already have the document.
            # if existing:
            #     print(f"\nSkipping '{paper_name}' as it is already in the database.")
            #     ## Update gist.
            #     parsed_list.append(paper_name)
            #     paper_list = list(set(paper_list) - set(parsed_list))
            #     gist_url = update_gist(gist_id, gist_filename, paper_list)
            #     continue

            ## Search content.
            try:
                new_doc = pu.search_arxiv_doc(paper_name)
            except Exception as e:
                print(f"\nFailed to search for '{paper_name}'. Skipping...")
                print(e)
                continue

            if new_doc is None:
                print(f"\nCould not find '{paper_name}' in Arxiv. Skipping...")
                continue

            new_meta = new_doc.metadata
            new_content = pu.preprocess_arxiv_doc(new_doc.page_content)
            title = new_meta["Title"]
            arxiv_code = new_meta["entry_id"].split("/")[-1]
            arxiv_code = re.sub(r"v\d+$", "", arxiv_code)

            ## Verify it's an LLM paper.
            is_llm_paper = vs.verify_llm_paper(new_content[:1500] + " ...[continued]...")
            if not is_llm_paper["is_related"]:
                print(f"\n'{paper_name}' - '{title}' is not a LLM paper. Skipping...")
                parsed_list.append(paper_name)
                paper_list = list(set(paper_list) - set(parsed_list))
                gist_url = update_gist(gist_id, gist_filename, paper_list)
                ## Store in nonllm_arxiv_text.
                pu.store_local(new_content, arxiv_code, "nonllm_arxiv_text", format="txt")
                continue

            ## Check if we have a summary locally.
            local_paper_codes = os.path.join(
                os.environ.get("PROJECT_PATH"), "data", "arxiv_text"
            )
            local_paper_codes = [
                f.split(".json")[0] for f in os.listdir(local_paper_codes)
            ]
            if arxiv_code in local_paper_codes:
                ## Update gist.
                print(f"\nFound '{paper_name}' - '{title}' locally. Skipping...")
                parsed_list.append(paper_name)
                paper_list = list(set(paper_list) - set(parsed_list))
                gist_url = update_gist(gist_id, gist_filename, paper_list)
                continue

            ## Store.
            pu.store_local(new_content, arxiv_code, "arxiv_text", format="txt")
            print(f"\nSummary for '{paper_name}' - '{title}' stored locally.")

            ## Update gist.
            parsed_list.append(paper_name)
            paper_list = list(set(paper_list) - set(parsed_list))
            gist_url = update_gist(gist_id, gist_filename, paper_list)

    if gist_url:
        print(f"Done! Updated queue gist URL: {gist_url}")
    print(cb)


if __name__ == "__main__":
    main()
