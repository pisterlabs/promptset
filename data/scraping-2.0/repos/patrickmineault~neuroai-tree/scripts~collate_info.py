import numpy as np
import pandas as pd
from transformers import pipeline


def authorships_to_string(authorships):
    names = [a["author"].get("display_name", "") for a in authorships]
    if len(names) > 5:
        return ", ".join(names[:5]) + ", et al."
    return ", ".join(names)


def authorships_to_string_unabridged(authorships):
    names = [a["author"].get("display_name", "") for a in authorships]
    return ", ".join(names)


def get_highlighter():
    qa_model = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad",
        revision="626af31",
    )
    question = (
        """What is biologically inspired by the brain, cortex, neuroscience or """
        """psychology, excluding deep or convolutional neural networks?"""
    )
    return qa_model, question.strip()


def highlight_abstracts(df):
    highlighter, question = get_highlighter()
    highlighted = []
    for abstract in df.abstract:
        try:
            highlight = highlighter(question, abstract)
            abstract_highlighted = (
                abstract[: highlight["start"]]
                + " **"
                + highlight["answer"]
                + "** "
                + abstract[highlight["end"] :]
            )
            highlighted.append(abstract_highlighted)
        except ValueError:
            # No answer found.
            highlighted.append(abstract)
    df["abstract_highlighted"] = highlighted
    return df


def get_journal_name(x):
    if (
        "source" in x
        and x["source"] is not None
        and "display_name" in x["source"]
    ):
        return x["source"]["display_name"]
    return ""


def main():
    df = pd.read_json("data/processed/works.jsonl", lines=True)
    df = df.rename(columns={"source": "origin"})

    df_ss = pd.read_json("data/processed/semantic_scholar.jsonl", lines=True)
    df_ss["ss_cited_by_count"] = df_ss["result"].map(
        lambda x: x["citationCount"]
    )
    df_ss["ssid"] = df_ss["result"].map(lambda x: x["paperId"])
    df_ss = df_ss[["id", "ss_cited_by_count", "ssid"]]

    # Do a left join on the paper ID
    df = df.merge(df_ss, left_on="id", right_on="id", how="left")

    # Drop bad rows
    df = df[~df["id"].duplicated()]
    df = df[~df["ssid"].duplicated()]
    df = df[df.title != "Title"]

    df["author_list"] = df.authorships.map(authorships_to_string)
    df["author_list_unabridged"] = df.authorships.map(
        authorships_to_string_unabridged
    )
    df["journal"] = df.primary_location.map(lambda x: get_journal_name(x))
    df["link"] = df["primary_location"].map(lambda x: x["landing_page_url"])

    # Get the classification from OpenAI
    df_class = pd.read_json(
        "data/processed/coarse_classification.jsonl", lines=True
    )
    df = df.merge(df_class, on="id")

    # Get the coarse classification from the keyword-based detection.
    df_class = pd.read_json("data/processed/categories.jsonl", lines=True)
    df = df.merge(df_class, on="id")

    cites = (df["oa_neuro_citations"].values >= 2) | (
        df["ss_neuro_citations"].values >= 2
    )
    keywords = df["keywords_found"].values >= 1
    manual = df["origin"] == "manual"
    df["reason"] = np.where(
        manual,
        "Manually added",
        np.where(
            cites & keywords,
            "Matched 1+ abstract keywords & cited 2+ neuro papers",
            np.where(
                keywords,
                "Matched 1+ abstract keywords",
                np.where(
                    cites,
                    "Cited 2+ neuro papers",
                    "Other",
                ),
            ),
        ),
    )

    df_all = df.copy()

    df = df[~df["openai_category"].isna()]

    assert df.shape[0] < 10000, "Too many papers!"

    df = highlight_abstracts(df)

    df = df[
        [
            "id",
            "ssid",
            "title",
            "publication_year",
            "journal",
            "link",
            "author_list",
            "author_list_unabridged",
            "cited_by_count",
            "openai_category",
            "abstract",
            "abstract_highlighted",
            "ss_cited_by_count",
            "oa_cited_journals",
            "ss_cited_journals",
            "reason",
        ]
    ]

    df = df[~df.id.duplicated()]

    # Save the final dataframe
    df.to_csv("data/processed/neuroai-works.csv", index=False)

    df_all = df_all[~df_all.id.duplicated()]
    df_all = df_all[
        [
            "id",
            "ssid",
            "title",
            "publication_year",
            "journal",
            "link",
            "author_list",
            "cited_by_count",
            "openai_category",
            "ss_cited_by_count",
            "oa_cited_journals",
            "ss_cited_journals",
            "reason",
        ]
    ]
    df_all.to_csv("data/processed/all-works.csv", index=False)


if __name__ == "__main__":
    main()
