"""Supporting functions for LLM meta-analysis."""


import logging

import bs4
import numpy as np
import pandas as pd
import tiktoken
from langchain.embeddings import (
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
    OpenAIEmbeddings,
)
from lxml import etree
from sklearn.neighbors import KDTree


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    log_file_name = "project.log"
    file_handler = logging.FileHandler(log_file_name)
    stream_handler = logging.StreamHandler()

    # Set logging level for handlers
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)

    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

    # Add formatters to handlers
    file_handler.setFormatter(file_formatter)
    stream_handler.setFormatter(stream_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


logger = setup_logger(__name__)


def count_tokens(text, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def assignments_to_df(data: dict[str, dict[str, list[str]]]) -> pd.DataFrame:
    """Load yaml-format mapping of label to doi into dataframe."""
    rows = []
    for node, dois in data.items():
        for doi in dois:
            rows.append({"node": node, "doi": doi})

    df = pd.DataFrame(rows)

    return df


def search_db(db, doi, query, k=10, min_tokens=10, min_score=1e9):
    """Semantic/similarity search on vector store."""
    _df = pd.DataFrame(
        [
            doc.metadata | {"score": score}
            for (doc, score) in db.similarity_search_with_score(
                query,
                k=k,
                filter={
                    "$and": [
                        {"doi": doi},
                        {"is_heading": False},
                        {"num_tokens": {"$gte": min_tokens}},
                    ]
                },
            )
        ]
    )

    if _df.empty:
        logger.info(f"No matches for [{query[:10]}]")
        return None

    return _df.query("score <= @min_score").loc[:, ["global_ordinal", "score"]]


def load_embeddings_model(embeddings_model, query_instruction=""):
    """Load a provided embeddings model."""
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True}

    if embeddings_model == "openai":
        embeddings = OpenAIEmbeddings()
    elif "BAAI" in embeddings_model:
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=embeddings_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            query_instruction=query_instruction,
        )
    else:
        try:
            embeddings = HuggingFaceInstructEmbeddings(
                model_name=embeddings_model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                query_instruction=query_instruction,
            )
            logger.debug(
                "Failed Instruct import, moving on to generic embeddings import."
            )
        except ImportError:
            embeddings = HuggingFaceEmbeddings(
                model_name=embeddings_model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )

    return embeddings


def preprocess_doc(
    db,
    doi,
    min_tokens=20,
    max_tokens=1000,
    literal_queries=None,
    semantic_queries=None,
):
    """Load all the context we'll use for examining a document.

    1. Pulls the per-sentence metadata into a df for convenience.
    2. Loads doi, journal, title up.
    3. Finds "best" sentences subject to token limit.

    Returns a dict to interpolate into prompts.
    """
    records = db.get(
        where={"doi": doi}, include=["metadatas", "documents", "embeddings"]
    )
    if not records:
        return None

    # Construct metadata dataframe.
    paper_data = (
        pd.DataFrame(records["metadatas"])
        .assign(
            embeddings=records["embeddings"],
            document=records["documents"],
            # head_id=lambda f: f.head_id.ffill().bfill(),
            heading=lambda f: f.text.where(f.is_heading, pd.NA).ffill(),
        )
        .sort_values("global_ordinal")
    )
    if paper_data.empty:
        logger.error(f"No records for paper [{doi}]")
        return

    journal = paper_data.journal_name.unique()[0]
    title = paper_data.paper_title.unique()[0]

    # Select sentences from paper to be included.
    if max_tokens > 0:
        selected_sentences = select_paper_sentences(
            db,
            paper_data,
            doi,
            min_tokens,
            max_tokens,
            literal_queries,
            semantic_queries,
        )
    else:
        # No token limit, just forward the whole paper.
        selected_sentences = paper_data.query("~is_heading")

    # Build markdown text for interpolation into prompt.
    _selected_md = []
    for g, sdf in selected_sentences.groupby("heading", sort=False):
        _selected_md += [f"- {g}:"]
        for t in sdf.itertuples():
            _selected_md += [f"    + {t.text}:"]
    selected_md = "\n".join(_selected_md)

    return {
        "doi": doi,
        "journal": journal,
        "title": title,
        "selected_md": selected_md,
    }


def _thin_sentences(df, max_tokens):
    """Reduce provided collection of sentences to `max_tokens`."""
    # We use the index here; ensure that it's clean to start.
    df = df.reset_index(drop=True)

    # Build kd-tree
    tree = KDTree([np.asarray(e) for e in df.embeddings.tolist()])

    logger.info(f"Thinning start: {df.num_tokens.sum()} tokens")
    while df["num_tokens"].sum() > max_tokens:
        # Query kd-tree for two closest points
        distances, indices = tree.query(
            [np.asarray(e) for e in df.embeddings.tolist()], k=2
        )
        min_distance_index = np.argmin(distances[:, 1])

        # Find the closest sentence pair in embeddings space
        closest_pair = indices[min_distance_index]

        # Remove the shorter sentence in terms of tokens, of the two closest embeddings.
        _df = df.loc[closest_pair].sort_values("num_tokens")
        index_to_remove = _df.index[0]

        # Remove from DataFrame and embeddings array
        df = df.drop(index=index_to_remove).reset_index(drop=True)

        # Rebuild kd-tree with remaining points
        tree = KDTree([np.asarray(e) for e in df.embeddings.tolist()])

    logger.info(f"Thinning stop: {df.num_tokens.sum()} tokens")

    return df


def select_paper_sentences(
    db, paper_data, doi, min_tokens, max_tokens, literal_queries, semantic_queries
):
    """Implement semantic + literal search and thinning."""
    matches = []

    # Look up literal keyword matches.
    for t in literal_queries:
        _matches = (
            paper_data.query("document.str.contains(@t)")
            .query("num_tokens >= @min_tokens")
            .loc[:, ["global_ordinal"]]
            .assign(score=1, source=f"literal-match-[{t}]")
        )
        matches.append(_matches)

    # Do semantic/similarity searches.
    for _ix, q in enumerate(semantic_queries):
        _matches = search_db(db, doi, q, min_tokens=min_tokens)
        if _matches is not None:
            matches.append(_matches.assign(source=f"semantic-match-{q[:10]}"))

    # Combine results from two searches.
    matches = (
        pd.concat(matches).sort_values(["global_ordinal", "source"])
        # Attach paper/sentence metadata, including embeddings.
        .merge(
            paper_data.loc[
                :, ["global_ordinal", "heading", "num_tokens", "text", "embeddings"]
            ],
            on="global_ordinal",
        )
    )

    logger.info(matches.groupby("source").num_tokens.sum())

    # Ignore score-related info.
    matches = (
        matches.drop(columns=["score", "source"])
        .drop_duplicates("text")
        .reset_index(drop=True)
    )

    return _thin_sentences(matches, max_tokens)


def process_xml_etree(xml_data: str) -> list[dict[str, str]]:
    """Process Grobid-emitted XML into chromadb-able document structures."""
    try:
        root = etree.fromstring(xml_data)
    except ValueError:
        root = etree.fromstring(xml_data.encode("utf8"))

    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    doi_element = root.find('.//tei:idno[@type="DOI"]', namespaces=ns)
    doi = doi_element.text if doi_element is not None else "No DOI Found"

    paper_title_element = root.find(
        './/tei:fileDesc//tei:title[@type="main"]', namespaces=ns
    )
    paper_title = (
        paper_title_element.text
        if paper_title_element is not None
        else "No Title Found"
    )

    journal_name_element = root.find(
        './/tei:monogr//tei:title[@type="main"]', namespaces=ns
    )
    journal_name = (
        journal_name_element.text
        if journal_name_element is not None
        else "No Journal Name Found"
    )

    shared_metadata = {
        "parsed_doi": doi,
        "paper_title": str(paper_title),
        "journal_name": str(journal_name),
    }

    # Process abstract.
    docs = [
        dict(
            page_content="Abstract",
            metadata=shared_metadata
            | {
                "is_heading": True,
                "num_tokens": count_tokens("Abstract"),
                "paragraph_id": "heading",
                "text": "Abstract",
            },
        )
    ]
    abstract_div = root.find(".//tei:profileDesc/tei:abstract/tei:div", namespaces=ns)
    if abstract_div is not None:
        for p in abstract_div.findall(".//tei:p", namespaces=ns):
            paragraph_id = p.get("{http://www.w3.org/XML/1998/namespace}id")
            for s in p.findall(".//tei:s", namespaces=ns):
                sentence_text = "".join(s.itertext()).strip()
                docs += [
                    dict(
                        page_content=sentence_text,
                        metadata=shared_metadata
                        | {
                            "is_heading": False,
                            "num_tokens": count_tokens(sentence_text),
                            "paragraph_id": paragraph_id,
                            "text": sentence_text,
                        },
                    )
                ]

    else:
        print("No abstract found.")

    # Process body of paper.
    divs = root.findall(".//tei:text/tei:body/tei:div", namespaces=ns)

    for div in divs:
        head_text = div.find(".//tei:head", namespaces=ns)
        section_heading = head_text.text if head_text is not None else "No Heading"
        docs += [
            dict(
                page_content=section_heading,
                metadata=shared_metadata
                | {
                    "is_heading": True,
                    "num_tokens": count_tokens(section_heading),
                    "paragraph_id": "heading",
                    "text": section_heading,
                },
            )
        ]

        for p in div.findall(".//tei:p", namespaces=ns):
            paragraph_id = p.get("{http://www.w3.org/XML/1998/namespace}id")
            for s in p.findall(".//tei:s", namespaces=ns):
                sentence_text = "".join(s.itertext()).strip()
                docs += [
                    dict(
                        page_content=sentence_text,
                        metadata=shared_metadata
                        | {
                            "is_heading": False,
                            "num_tokens": count_tokens(sentence_text),
                            "paragraph_id": paragraph_id,
                            "text": sentence_text,
                        },
                    )
                ]

    # Add global ordinal.
    for ix, e in enumerate(docs):
        e["metadata"]["global_ordinal"] = ix

    return docs


def process_xml_bs4(xml_data: str) -> list[dict[str, str]]:
    """Process the XML file from Grobid.

    This is derived from LangChain.
    """

    soup = bs4.BeautifulSoup(xml_data, "xml")

    title = soup.find_all("title")[0].text
    logger.debug(f"Found {title=}")

    doi_tag = soup.find("idno", {"type": "DOI"})
    if doi_tag is not None:
        doi = doi_tag.text
        logger.debug(f"The DOI is: {doi}")
    else:
        doi = None
        logger.debug("DOI tag not found.")

    # Find the <monogr> tag
    monogr_tag = soup.find("monogr")

    # Within <monogr>, find the <title> tag with the specific attributes
    journal_name = None
    if monogr_tag is not None:
        title_tag = monogr_tag.find("title", {"level": "j", "type": "main"})
        if title_tag is not None:
            journal_name = title_tag.text
            logger.debug(f"The journal name is: {journal_name}")
        else:
            logger.debug("Journal name not found.")
    else:
        logger.debug("<monogr> tag not found.")

    shared_metadata = {
        "parsed_doi": doi,
        "paper_title": str(title),
        "journal_name": str(journal_name),
    }

    text = "Abstract"
    docs = [
        dict(
            page_content=text,
            metadata=shared_metadata
            | {
                "is_heading": True,
                "num_tokens": count_tokens(text),
                "text": text,
            },
        )
    ]
    for ix, sentence in enumerate(soup.find("abstract").find_all("s")):
        text = sentence.text
        doc = dict(
            page_content=text,
            metadata=shared_metadata
            | {
                "is_heading": False,
                "num_tokens": count_tokens(text),
                "text": text,
                "section_ordinal": ix,
                "head_id": "0",
            },
        )
        docs.append(doc)

    # Loop through all divs
    for div in soup.find("body").find_all("div"):
        head = div.find("head")
        head_id = None

        # Check if this div has a head element
        if head:
            head_id = head.get("n")
            head_text = head.text
            doc = dict(
                page_content=head.text,
                metadata=shared_metadata
                | {
                    "is_heading": True,
                    "num_tokens": count_tokens(head_text),
                    "text": head_text,
                    "head_id": head_id,
                },
            )
            docs.append(doc)

        for ix, s in enumerate(div.find_all("s")):
            text = s.text
            doc = dict(
                page_content=text,
                metadata=shared_metadata
                | {
                    "is_heading": False,
                    "num_tokens": count_tokens(text),
                    "text": text,
                    "section_ordinal": ix,
                },
            )
            if head_id is not None:
                doc["metadata"]["head_id"] = head_id
            docs.append(doc)

    # Add global ordinal.
    for ix, e in enumerate(docs):
        e["metadata"]["global_ordinal"] = ix

    return docs


process_xml = process_xml_etree
