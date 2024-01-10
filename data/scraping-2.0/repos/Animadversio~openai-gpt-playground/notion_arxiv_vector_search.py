import os
import numpy as np
from os.path import join
from notion_tools import print_entries
import pickle as pkl
import arxiv
import questionary
#%%
abstr_embed_dir = "/Users/binxuwang/Library/CloudStorage/OneDrive-HarvardUniversity/openai-emb-database/Embed_arxiv_abstr"
database_catalog = {# "diffusion_7k": "arxiv_embedding_arr_diffusion_7k.pkl",
                    # "LLM_5k": "arxiv_embedding_arr_LLM_5k.pkl",
                    "diffusion_10k": "arxiv_embedding_arr_diffusion_10k.pkl",
                    "LLM_18k": "arxiv_embedding_arr_LLM_18k.pkl",
                    "GAN_6k": "arxiv_embedding_arr_GAN_6k.pkl",
                    "VAE_2k": "arxiv_embedding_arr_VAE_2k.pkl",
                    "flow_100": "arxiv_embedding_arr_flow_100.pkl",
                    "normflow_800": "arxiv_embedding_arr_normflow_800.pkl",
}
# database_name = "diffusion_7k"
database_name = questionary.select("Select database to browse:", choices=list(database_catalog.keys())+["All"]).ask()
if not (database_name == "All"):
    database_file = database_catalog[database_name]
    embed_arr, paper_collection = pkl.load(open(join(abstr_embed_dir, database_file), "rb"))
else:
    embed_arr = []
    paper_collection = []
    for database_file in database_catalog.values():
        embed_arr_cur, paper_collection_cur = pkl.load(open(join(abstr_embed_dir, database_file), "rb"))
        embed_arr.append(embed_arr_cur)
        paper_collection.extend(paper_collection_cur)
    embed_arr = np.concatenate(embed_arr, axis=0)

assert embed_arr.shape[0] == len(paper_collection)
print(f"Loaded {embed_arr.shape[0]} papers from {database_name}, embed shape {embed_arr.shape}")
#%%
from notion_client import Client
from notion_tools import print_entries
import arxiv
import questionary
import textwrap
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory, FileHistory
import openai
client = openai.OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

history = FileHistory("notion_abstract_history.txt")
session = PromptSession(history=history)

MAX_RESULTS = 25
database_id = "d3e3be7fc96a45de8e7d3a78298f9ccd"
notion = Client(auth=os.environ["NOTION_TOKEN"])

def fetch_K_vector_neighbor(cossim, paper_collection, K=10, offset=0):
    sort_idx = np.argsort(cossim)
    sort_idx = sort_idx[::-1]
    sort_idx = sort_idx[offset:offset+K]
    return [paper_collection[idx] for idx in sort_idx]


def print_arxiv_entry(paper: arxiv.arxiv.Result):
    title = paper.title
    authors = [author.name for author in paper.authors]
    pubyear = paper.published
    abstract = paper.summary
    arxiv_id = paper.entry_id.split("/")[-1]
    abs_url = paper.entry_id
    print(f"[{arxiv_id}] {title}")
    print("Authors:", ", ".join(authors))
    print("Published:", pubyear.date().isoformat())
    print("Abstract:")
    print(textwrap.fill(abstract, width=100))
    print("comments:", paper.comment)
    print("URL:", abs_url)



def arxiv_entry2page_blocks(paper: arxiv.arxiv.Result):
    title = paper.title
    authors = [author.name for author in paper.authors]
    pubyear = paper.published
    abstract = paper.summary
    arxiv_id = paper.entry_id.split("/")[-1]
    abs_url = paper.entry_id
    page_prop = {
        'Name': {
            "title": [
                {
                    "text": {
                        "content": f"[{arxiv_id}] {title}"
                    }
                }],
        },
        "Author": {
            "multi_select": [
                {'name': name} for name in authors
            ]
        },
        'Publishing/Release Date': {
            'date': {'start': pubyear.date().isoformat(), }
        },
        'Link': {
            'url': abs_url
        }
    }
    content_block = [{'quote': {"rich_text": [{"text": {"content": abstract}}]}},
                     {'heading_2': {"rich_text": [{"text": {"content": "Related Work"}}]}},
                     {'paragraph': {"rich_text": [{"text": {"content": ""}}]}},
                     {'heading_2': {"rich_text": [{"text": {"content": "Techniques"}}]}},
                     {'paragraph': {"rich_text": [{"text": {"content": ""}}]}},
                     ]
    return page_prop, content_block


def arxiv_entry2page(database_id, paper: arxiv.arxiv.Result):
    page_prop, content_block = arxiv_entry2page_blocks(paper)
    new_page = notion.pages.create(parent={"database_id": database_id}, properties=page_prop)
    notion.blocks.children.append(new_page["id"], children=content_block)
    return new_page["id"], new_page



def blocks2text(blocks):
    if "results" in blocks:
        blocks = blocks["results"]
    for block in blocks:
        if block["type"] == "paragraph":
            for parts in block["paragraph"]["rich_text"]:
                print(textwrap.fill(parts["plain_text"], width=100))

        elif block["type"] == "heading_2":
            for parts in block["heading_2"]["rich_text"]:
                print(textwrap.fill(parts["plain_text"], width=100))

        elif block["type"] == "quote":
            for parts in block["quote"]["rich_text"]:
                print(textwrap.fill(parts["plain_text"], width=100))
        else:
            print(block["type"])


def add_to_notion(paper: arxiv.arxiv.Result):
    title = paper.title
    arxiv_id = paper.entry_id.split("/")[-1]
    # check if entry already exists in Notion database
    results_notion = notion.databases.query(database_id=database_id,
                                            filter={"property": "Link", "url": {"contains": arxiv_id}})
    if len(results_notion["results"]) == 0:
        print(f"Adding entry paper {arxiv_id}: {title}")
        page_id, page = arxiv_entry2page(database_id, paper)
        print(f"Added entry {page_id} for arxiv paper {arxiv_id}: {title}")
        print_entries([page], print_prop=("url",))
    else:
        print_entries(results_notion, print_prop=("url",))
        print("Entry already exists as above. Exiting.")
        for page in results_notion["results"]:
            print_entries([page], print_prop=("url",))
            try:
                blocks = notion.blocks.children.list(page["id"])
                blocks2text(blocks)
            except Exception as e:
                print(e)


while True:
    try:
        query = session.prompt("Enter query str to search arxiv database: ",
                               multiline=False)
        response_query = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embed = np.array(response_query.data[0].embedding)
        sim = embed_arr @ query_embed
        cossim = (sim / np.linalg.norm(embed_arr, axis=1)
                  / np.linalg.norm(query_embed))
        #%%
        offset_cur = 0
        results_arxiv = fetch_K_vector_neighbor(cossim, paper_collection, K=MAX_RESULTS, offset=offset_cur)
        last_selection = None  # last selected result to highlight
        while True:
            # looping of results and pages, navigating through search results
            choices = [f"{i + 1}: [{paper.entry_id.split('/')[-1]}] {paper.title} "
                        for i, paper in enumerate(results_arxiv)]
            if len(results_arxiv) == MAX_RESULTS:
                choices.append("0: Next page")
            if offset_cur > 0:
                choices.append("-1: Prev page")
            selection = questionary.select("Select paper:", choices=choices, default=None if last_selection is None
                                           else choices[last_selection]).ask()
            selection = int(selection.split(":")[0])
            if selection == 0:
                offset_cur += MAX_RESULTS
                results_arxiv = fetch_K_vector_neighbor(cossim, paper_collection, K=MAX_RESULTS, offset=offset_cur)
                continue
            if selection == -1:
                offset_cur -= MAX_RESULTS
                results_arxiv = fetch_K_vector_neighbor(cossim, paper_collection, K=MAX_RESULTS, offset=offset_cur)
                continue
            else:
                paper = results_arxiv[int(selection) - 1]
                last_selection = int(selection) - 1
                print_arxiv_entry(paper)
                if questionary.confirm("Add this entry?").ask():
                    # Add the entry if confirmed
                    add_to_notion(paper)
                # if questionary.confirm("Back to the list").ask():
                #     continue
                # else:
                #     break

    except Exception as e:
        continue

#%%