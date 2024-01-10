import json
from typing import Dict, List

from langchain.docstore.document import Document

from inference.prompts import (
    extra_info_chain,
    get_style_critique_chain,
    group_chain,
    new_sum_chain,
    purpose_chain,
    synth_chain,
    synth_combo_chain,
    tid_chain,
)


def generate_initial_bullets(news_article: str, persona: str) -> str:
    """Generate initial bullets."""
    # Extract interesting tidbits
    tid_dict = {"article": news_article, "profession": persona}
    tid_res = tid_chain(tid_dict)["text"]

    # Extract primary interest/purpose of text from persona perspective
    purpose_res = purpose_chain(tid_dict)["text"]

    # Group interesting parts into themes
    article_dict = {"article": news_article, "points": tid_res, "profession": persona}
    order = group_chain(article_dict)["text"]

    # Organize into bulleted summary
    article_dict = {
        "article": news_article,
        "points": tid_res,
        "themes": order,
        "prof_purpose": purpose_res,
    }

    constitutional_chain = get_style_critique_chain(persona, new_sum_chain)
    bullet_output = constitutional_chain(article_dict)["output"]

    return bullet_output


def generate_extra_info_bullets(
    bullets_to_synthesize: List[str],
    docs_to_include_for_bullets: List[List[Document]],
    _: str,
) -> List[str]:
    """Generate extra info bullets."""
    extra_info_bullets = []
    for i, text in enumerate(bullets_to_synthesize):
        val = "\n".join(
            ["- " + doc.page_content for doc in docs_to_include_for_bullets[i]]
        )
        extra_info_dict = {
            "first": bullets_to_synthesize[i],
            "second": val,
        }
        extra_res = extra_info_chain(extra_info_dict)["text"]
        extra_dict = json.loads(extra_res)
        if extra_dict["extra_info_needed"] == "YES":
            extra_info_bullets.append(extra_dict["extra_info"])
    return extra_info_bullets


def generate_synthesis_bullets(
    relation_dict: Dict, doc_dict: Dict, persona: str
) -> List[str]:
    """Generate synthesis bullets."""
    synth_results = []
    for rel_key in relation_dict.keys():
        relevant_to_prev = "\n".join(["- " + text for text in relation_dict[rel_key]])
        synth_dict = {
            "first": relevant_to_prev,
            "second": rel_key,
            "profession": persona,
        }
        synth_res = synth_chain(synth_dict)["text"]
        link = doc_dict[rel_key].metadata["link"]
        synth_results.append(synth_res + f" (Source: {link})")

    if len(synth_results) == 0:
        return []

    synth_combo_dict = {
        "bullets": "\n".join(["- " + result for result in synth_results]),
        "personality": persona
    }
    constitutional_chain = get_style_critique_chain(persona, synth_combo_chain)
    synth_bullets = constitutional_chain(synth_combo_dict)["output"]
    ind_synths = synth_bullets.split("\n")
    return ind_synths


def separate_bullet_output(bullet_output: str) -> List[str]:
    """Separate bullet output into bullets."""
    # Split bulleted output up
    bullets = bullet_output.split("\n")
    cleaned_bullets = []
    for b in bullets:
        b = b.strip("-").strip()
        cleaned_bullets.append(b)

    return cleaned_bullets
