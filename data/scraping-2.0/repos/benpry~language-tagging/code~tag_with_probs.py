"""
This file contains functions that get probabilities of each tag (rather than just the most likely tag) for each sentence.
"""
import os
from openai import OpenAI
from prompts import chat_prompt_templates
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from pyprojroot import here


def format_messages(prompt, test_sentence):
    """
    Turn a LangChain prompt into a list of dictionaries to pass to openai.
    """
    messages = prompt.format_prompt(test_sentence=test_sentence).to_messages()
    message_dicts = [msg.dict() for msg in messages]
    for msg in message_dicts:
        del msg["additional_kwargs"]
        if msg["type"] == "ai":
            msg["type"] = "assistant"
        if msg["type"] == "human":
            msg["type"] = "user"
        msg["role"] = msg["type"]
        del msg["type"]
        if "example" in msg:
            del msg["example"]
    return message_dicts


openai = OpenAI()


tag_first_tokens = {
    "abstract": {"abstract": "abstract", "con": "concrete", "ign": "ignorance"},
    "policy": {"not": "not policy", "policy": "policy"},
    "dynamics": {"not": "not dynamics", "d": "dynamics"},
    "valence": {"win": "winning", "losing": "losing", "neutral": "neutral"},
}


def get_tag_probabilities(test_sentence, prompt, tag_type, model="gpt-3.5-turbo"):
    """
    Get the probability of each tag for a given sentence.
    """
    response = openai.chat.completions.create(
        model=model,
        messages=format_messages(prompt, test_sentence),
        logprobs=True,
        top_logprobs=len(tag_first_tokens[tag_type]),
        n=1,
        max_tokens=1,
        temperature=0,
    )

    log_prob_dict = {}
    logprobs = response.choices[0].logprobs.content[0].top_logprobs
    print(logprobs)
    for log_prob in logprobs:
        if log_prob.token in tag_first_tokens[tag_type]:
            log_prob_dict[tag_first_tokens[tag_type][log_prob.token]] = log_prob.logprob

    # if any of the tags are missing, set their probability to 0
    for tag in tag_first_tokens[tag_type].values():
        if tag not in log_prob_dict:
            print(f"Warning: {tag} not in log prob dict")
            log_prob_dict[tag] = -np.inf

    return log_prob_dict


def assign_tag_from_probs(log_prob_dict, tag_type):
    """
    Given log probabilities, assign a tag.
    """

    # tag for abstraction, biasing the tagging toward concreteness
    if tag_type == "abstract":
        p_ignorance, p_abstract, p_concrete = np.exp(
            [
                log_prob_dict["ignorance"],
                log_prob_dict["abstract"],
                log_prob_dict["concrete"],
            ]
        )
        if p_ignorance > p_abstract and p_ignorance > p_concrete:
            return "ignorance"
        elif p_abstract > 0.01:
            return "abstract"
        elif p_concrete > 0:
            return "concrete"
        else:
            return "unknown"

    # tag for policy, biasing toward not policy
    elif tag_type == "policy":
        if np.exp(log_prob_dict["policy"]) > 0.6:
            return "policy"
        elif np.exp(log_prob_dict["not policy"]) > 0.4:
            return "not policy"
        else:
            return "unknown"

    # tag for dynamics, biasing slightly toward not dynamics
    elif tag_type == "dynamics":
        if np.exp(log_prob_dict["dynamics"]) > 0.45:
            return "dynamics"
        elif np.exp(log_prob_dict["not dynamics"]) > 0.55:
            return "not dynamics"
        else:
            return "unknown"

    # tag for valence, biasing slightly away from neutral
    elif tag_type == "valence":
        if np.exp(log_prob_dict["neutral"]) > 0.5:
            return "neutral"
        elif np.exp(log_prob_dict["winning"]) > 0.3:
            return "winning"
        elif np.exp(log_prob_dict["losing"]) > 0.3:
            return "losing"
        else:
            return "unknown"


def tag_sentences_prob(sentences):
    rows = []
    for sentence in sentences:
        row = {"sentence": sentence}
        for tag_type in tag_first_tokens.keys():
            log_prob_dict = get_tag_probabilities(
                sentence, chat_prompt_templates[tag_type], tag_type
            )
            for val, log_prob in log_prob_dict.items():
                row[f"{tag_type}_{val}"] = log_prob
            tag = assign_tag_from_probs(log_prob_dict, tag_type)
            row[f"{tag_type}"] = tag
        rows.append(row)

    return pd.DataFrame(rows)


parser = ArgumentParser()

parser.add_argument(
    "--messages_path",
    type=str,
    default="data/raw-data/tagged_annotated_sentences.csv",
)
parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")

if __name__ == "__main__":
    args = parser.parse_args()
    prompt_types = chat_prompt_templates.keys()

    messages_path = here() / args.messages_path

    # Load in the messages
    df_messages = pd.read_csv(messages_path)

    sentences = df_messages["sentence"].drop_duplicates()

    print(f"tagging {len(sentences)} sentences...")
    df_tagged = tag_sentences_prob(sentences)
    df_tagged.to_csv(here(f"data/tagged_sentences_prob.csv"), index=False)
