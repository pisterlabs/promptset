import os
import re, json
import arxiv
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import dotenv
import ast

from langchain.document_loaders import ArxivLoader

dotenv.load_dotenv()

PROJECT_PATH = os.environ.get("PROJECT_PATH")
DATA_PATH = os.path.join(PROJECT_PATH, "data")

db_params = {
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASS"],
    "host": os.environ["DB_HOST"],
    "port": os.environ["DB_PORT"],
}

summary_col_mapping = {
    "arxiv_code": "arxiv_code",
    "main_contribution_headline": "contribution_title",
    "main_contribution_description": "contribution_content",
    "takeaways_headline": "takeaway_title",
    "takeaways_description": "takeaway_content",
    "takeaways_example": "takeaway_example",
    "category": "category",
    "novelty_score": "novelty_score",
    "novelty_analysis": "novelty_analysis",
    "technical_score": "technical_score",
    "technical_analysis": "technical_analysis",
    "enjoyable_score": "enjoyable_score",
    "enjoyable_analysis": "enjoyable_analysis",
}

llm_terms = ["language", "llm", "artificial intelligence", "transformer"]

##################
## TXT ANALYSIS ##
##################
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), use_idf=False)


def tfidf_similarity(title1, title2, fitted=False):
    """Compute cosine similarity of TF-IDF representation between 2 strings."""
    title1 = preprocess(title1)
    title2 = preprocess(title2)
    if not fitted:
        vectors = vectorizer.fit_transform([title1, title2])
    else:
        vectors = vectorizer.transform([title1, title2])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]


def compute_optimized_similarity(data_title, titles):
    """Multithreading TF-IDF similarity computation."""
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(tfidf_similarity, data_title, title, True)
            for title in titles
        ]
    return [future.result() for future in futures]


def dict_similarity_matrix(doc_dict, ignore_columns=["Published"]):
    """Compute similarity matrix (DF) for elements in a dictionary."""
    df = pd.DataFrame.from_dict(doc_dict, orient="index").T
    df = df[[c for c in df.columns if not c.endswith("_score")]]
    df.drop(columns=ignore_columns, inplace=True)

    num_text_columns = len(df.columns)
    similarity_matrix = np.zeros((num_text_columns, num_text_columns))

    for i, col1 in enumerate(df.columns):
        for j, col2 in enumerate(df.columns):
            if i >= j:
                continue
            a = str(df[col1].values[0])
            b = str(df[col2].values[0])
            if len(a) < 10 or len(b) < 10:
                continue
            similarity_score = tfidf_similarity(a, b)
            similarity_matrix[i, j] = similarity_score
            similarity_matrix[j, i] = similarity_score

    similarity_df = pd.DataFrame(
        similarity_matrix, index=df.columns, columns=df.columns
    )

    return similarity_df


def get_high_similarity_pairs(similarity_df, x):
    """
    Returns a list of tuples containing pairs of columns and their
    similarity score where the similarity score is greater than x.
    Input is a similarity DF.
    """
    high_similarity_pairs = []
    cols = similarity_df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if similarity_df.iloc[i, j] > x:
                high_similarity_pairs.append(
                    ((cols[i], cols[j]), similarity_df.iloc[i, j])
                )

    return high_similarity_pairs


def classify_llm_paper(paper_content):
    """Check if a paper is a language model paper."""
    ##ToDo: Replace with LLM.
    keywords = [
        "language model",
        "llm",
        "transformer",
        "gpt",
        "bert",
        "attention",
        "encoder-decoder",
        "agent",
    ]
    skip_keywords = [
        "stable diffusion",
        "video diffusion",
        "image diffusion",
        "diffusion transformer",
        "image generation",
        "video generation",
        "text-to-image",
        "text-to-video",
        "text to image",
        "text to video",
    ]

    result = (any([k in paper_content.lower() for k in keywords])) and (
        not any([k in paper_content.lower() for k in skip_keywords])
    )
    return result


#####################
## LOCAL DATA MGMT ##
#####################


def get_local_arxiv_codes(directory="arxiv_text", format=".txt"):
    """Get a list of local Arxiv codes."""
    local_paper_codes = os.path.join(PROJECT_PATH, "data", directory)
    local_paper_codes = [
        f.split(format)[0] for f in os.listdir(local_paper_codes) if f.endswith(format)
    ]
    return local_paper_codes


def store_local(data, arxiv_code, data_path, relative=True, format="json"):
    """Store data locally."""
    if relative:
        data_path = os.path.join(DATA_PATH, data_path)
    if format == "json":
        with open(os.path.join(data_path, f"{arxiv_code}.json"), "w") as f:
            json.dump(data, f)
    elif format == "txt":
        with open(os.path.join(data_path, f"{arxiv_code}.txt"), "w") as f:
            f.write(data)
    else:
        raise ValueError("Format not supported.")


def load_local(arxiv_code, data_path, relative=True, format="json"):
    """Load data locally."""
    if relative:
        data_path = os.path.join(PROJECT_PATH, "data", data_path)
    if format == "json":
        with open(os.path.join(data_path, f"{arxiv_code}.json"), "r") as f:
            return json.load(f)
    elif format == "txt":
        with open(os.path.join(data_path, f"{arxiv_code}.txt"), "r") as f:
            return f.read()
    else:
        raise ValueError("Format not supported.")


#####################
## DATA PROCESSING ##
#####################
def reformat_text(doc_content):
    """Clean and simplify text string."""
    # content = doc_content.replace("-\n", "")
    # content = re.sub(r"(?<!\n)\n(?!\n)", " ", content)
    # content = re.sub(" +", " ", content)
    content = doc_content.replace("<|endoftext|>", "|endoftext|")
    return content


def format_paper_summary(summary_row):
    """Format a paper summary for display."""
    title = summary_row["title"]
    date = summary_row["published"].strftime("%B %d, %Y")
    arxiv_code = summary_row["arxiv_code"]
    citations = int(summary_row["citation_count"])
    summary = summary_row["recursive_summary"] if summary_row["recursive_summary"] else summary_row["summary"]
    main_contribution = summary_row["contribution_content"]
    takeaways = summary_row["takeaway_content"]
    arxiv_comments = summary_row["arxiv_comment"]
    arxiv_comments = "\n\n*" + arxiv_comments + "*" if arxiv_comments else ""
    return f"### {title}\n*({date} / arxiv_code:{arxiv_code} / {citations} citations)*{arxiv_comments}\n\n**Summary:**\n{summary}\n\n**Main Contribution:**\n{main_contribution}\n\n**Takeaways:**\n{takeaways}\n\n-------------\n\n"


def numbered_to_bullet_list(list_str: str):
    """Convert a numbered list to a bullet list."""
    list_str = re.sub(r"^\d+\.", r"-", list_str, flags=re.MULTILINE).strip()
    list_str = list_str.replace("</|im_end|>", "").strip()
    return list_str


def preprocess(text):
    """Clean and simplify text string."""
    text = "".join(c.lower() if c.isalnum() else " " for c in text)
    return text


def convert_string_to_dict(s):
    try:
        # Try to convert the string representation of a dictionary to an actual dictionary
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return s


def convert_innert_dict_strings_to_actual_dicts(data):
    if isinstance(data, str):
        return convert_string_to_dict(data)
    elif isinstance(data, dict):
        for key in data:
            data[key] = convert_innert_dict_strings_to_actual_dicts(data[key])
        return data
    elif isinstance(data, list):
        for i, item in enumerate(data):
            data[i] = convert_innert_dict_strings_to_actual_dicts(item)
        return data
    else:
        return data


def is_arxiv_code(s):
    """Check if a string is an Arxiv code."""
    pattern = re.compile(r"^\d{4}\.\d+$")
    return bool(pattern.match(s))


def flatten_dict(d, parent_key="", sep="_"):
    """Flatten a nested dictionary."""
    items = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def transform_flat_dict(flat_data, mapping):
    """Rename and drop columns from a flattened dictionary."""
    return {mapping[k]: flat_data[k] for k in mapping if k in flat_data}


#################
## ARXIV TOOLS ##
#################
def search_arxiv_doc(paper_name):
    """Search for a paper in Arxiv and return the most similar one."""
    is_code = is_arxiv_code(paper_name)
    max_docs = 1
    abs_check = True
    if not is_code:
        max_docs = 3
        abs_check = False
        paper_name = preprocess(paper_name)
    docs = ArxivLoader(
        query=paper_name,
        doc_content_chars_max=1000000,
        load_all_available_meta=True,
        load_max_docs=max_docs,
    ).load()

    if len(docs) == 0:
        return None

    if abs_check:
        ## Arxiv code must match exactly.
        arxiv_code = docs[0].metadata["entry_id"].split("/")[-1]
        if paper_name not in arxiv_code:
            return None
    else:
        ## Title must be highly similar.
        docs = sorted(
            docs,
            key=lambda x: tfidf_similarity(paper_name, x.metadata["Title"]),
            reverse=True,
        )
        new_title = docs[0].metadata["Title"]
        title_sim = tfidf_similarity(paper_name, new_title)
        if title_sim < 0.9:
            return None
    ## check if any of the language terms occur.
    # if not any([term in docs[0].page_content.lower() for term in llm_terms]):
    #     print("Not a language paper.")
    #     return None
    return docs[0]


def preprocess_arxiv_doc(doc_content, token_encoder=None, max_tokens=None, remove_references=True):
    """Preprocess an Arxiv document."""
    # doc_content = reformat_text(doc_content)
    doc_content = doc_content.replace("<|endoftext|>", "|endoftext|")

    if remove_references:
        if len(doc_content.split("References")) == 2:
            doc_content = doc_content.split("References")[0]

    if token_encoder:
        ntokens_doc = len(token_encoder.encode(doc_content))
        print(f"Number of tokens: {ntokens_doc}")
        if ntokens_doc > max_tokens:
            doc_content = doc_content[: int(max_tokens * 3)] + "... [truncated]"

    return doc_content


def get_arxiv_info(arxiv_code: str, title: Optional[str] = None):
    """Search article in Arxiv by name and retrieve meta-data."""
    search = arxiv.Search(
        query=arxiv_code, max_results=40, sort_by=arxiv.SortCriterion.Relevance
    )
    res = list(search.results())
    arxiv_meta = None
    if len(res) > 0:
        if title:
            ## Sort by title similarity.
            res = sorted(
                res, key=lambda x: tfidf_similarity(title, x.title), reverse=True
            )
            new_title = res[0].title
            title_sim = tfidf_similarity(title, new_title)
            if title_sim > 0.7:
                arxiv_meta = res[0]
        else:
            arxiv_meta = res[0]
    return arxiv_meta


def process_arxiv_data(data):
    """Transform the arxiv data for database insertion."""
    data = {k.lower(): v for k, v in data.items()}
    flat_data = flatten_dict(data)
    desired_fields = [
        "id",
        "updated",
        "published",
        "title",
        "summary",
        "authors",
        "arxiv_comment",
    ]
    filtered_data = {k: flat_data[k] for k in desired_fields if k in flat_data}
    filtered_data["arxiv_code"] = filtered_data.pop("id").split("/")[-1].split("v")[0]
    author_names = [author["name"] for author in filtered_data["authors"]]
    filtered_data["authors"] = ", ".join(author_names)
    filtered_data["authors"] = filtered_data["authors"][:1000]
    filtered_data["title"] = filtered_data["title"].replace("\n ", "")
    filtered_data["summary"] = filtered_data["summary"].replace("\n", " ")
    if "arxiv_comment" in filtered_data:
        filtered_data["arxiv_comment"] = filtered_data["arxiv_comment"].replace(
            "\n ", ""
        )
    return filtered_data


def get_semantic_scholar_info(arxiv_code):
    """Search article in Semantic Scholar by Arxiv code and retrieve meta-data."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_code}?fields=title,citationCount,influentialCitationCount,tldr,venue"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None


def check_if_exists(paper_name, existing_paper_names, existing_paper_ids):
    """Check if arxiv ID has exact match in existing papers or a very similar title."""
    if is_arxiv_code(paper_name):
        if paper_name in existing_paper_ids:
            return True
        else:
            return False
    else:
        pre_similarity = max(
            [tfidf_similarity(paper_name, t) for t in existing_paper_names]
        )
        if pre_similarity > 0.9:
            return True
        else:
            return False


##################
## GIST RELATED ##
##################
def fetch_queue_gist(gist_id, gist_filename="llm_queue.txt"):
    """Fetch the queue of papers to be reviewed from a GitHub gist."""
    response = requests.get(f"https://api.github.com/gists/{gist_id}")
    paper_list = None

    if response.status_code == 200:
        gist = response.json()
        paper_url = gist["files"][gist_filename]["raw_url"]
        response = requests.get(paper_url)
        if response.status_code == 200:
            paper_list = response.text.split("\n")
            paper_list = [p.strip() for p in paper_list if len(p.strip()) > 0]
            paper_list = list(set(paper_list))

    return paper_list


def update_gist(
    token: str,
    gist_id: str,
    gist_filename: str,
    gist_description: str,
    gist_content: str,
):
    """Upload a text file as a GitHub gist."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    params = {
        "description": gist_description,
        "files": {gist_filename: {"content": gist_content}},
    }
    response = requests.patch(
        f"https://api.github.com/gists/{gist_id}",
        headers=headers,
        data=json.dumps(params),
    )

    if response.status_code == 200:
        print(f"Gist {gist_filename} updated successfully.")
        return response.json()["html_url"]
    else:
        print(f"Failed to update gist. Status code: {response.status_code}.")
        return None
