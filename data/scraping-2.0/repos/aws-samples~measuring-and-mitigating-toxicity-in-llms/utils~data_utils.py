#!/usr/bin/env python3

import pandas as pd
import ast
from convokit import Corpus, download
import gc
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
import spacy
import random
import better_profanity
import os
from typing import Dict

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

_reddit_test_string = """I do like being a weirdo and a fucking asshole, so I'm glad your loser self has decided to fucking stop being a level-headed sissy, finally grew a fuckin pair of balls and asked! I can fucking cuss up a damn storm that'll make little pansies cry their faggoty bitch ass out! Seriously though, are you such a retarded idiot that you can't fucking figure this shit out? But to fucking answer your fucking question, profanity is what I'm fucking doing right now, using words that "polite society" doesn't have the balls to say, or those sons-of-bitches think are too fucking crude. Yes, "fuck" is a very versatile word which is why I'm using the fuck out of it, but there are other words like Marissa Pitts have fucking said, although some are just minor put downs or even some shit that doesn't fucking make a goddamn bit of sense... "Cult"? Really? Also it depends on the fucking culture, like Brit motherfuckers (AKA limeys) like to say cunt like they're fucking greeting their grandma and shit, but in the US it's as bad as calling a black dude a nigger, and you can REALLY fucking piss a black girl off by calling her a fucking "nigger cunt". Likewise, the fucking Brits can turn around and say "wanker", "quim", "pillock", "gobshite", "twit", and bloody 'ell, even "pony" for "pony and trap", rhyming slang for "crap". What about the fucking Germans? Sure, you can fucking call a kraut a nazi, but you can fucking say some weird ass shit like they like warm showers (warmduscher, or someone that's a fucking pansy because they can't shower with cold water... Hell, I'm a kaltduscher when I fucking work the shit out of myself at the fucking gym or even when I see a sexy bitch and I can't fuck her), a dumbass that irons their fucking socks (sockenbügler, because they're too obsessed with looking neat), someone who waves at the Teletubbies (Teletubbyzurückwinker, or a fucking retard), and all that shit. Does all this shit answer your fucking question?"""


def _get_metric_spec():
    """
    Rail spec template that applies filter upon fail condition.
    """
    text = """
    <rail version="0.1">

    <output>
        <string
            name="summarize_statement"
            format="is-profanity-free"
            on-fail-is-profanity-free="filter"
        />
    </output>

    <prompt>
    ${statement_to_be_summarized}
    </prompt>

    </rail>
    """
    return text


def _get_keyword_free_spec():
    """
    Rail spec template that applies fix upon fail condition.
    """
    text = """
    <rail version="0.1">

    <output>
        <string
            name="summarize_statement"
            format="is-keyword-free"
            on-fail-is-keyword-free="fix"
        />
    </output>

    <prompt>
    ${statement_to_be_summarized}
    </prompt>

    </rail>
    """
    return text


def _create_chunks(sample, CHUNK_LENGTH):
    """
    Splits a given text into chunks of a specified length and adds metadata to each chunk.
    """
    chunks = []
    # loop over entire text in steps of chunk size
    for c, i in enumerate(range(0, len(sample["dialogue"]), CHUNK_LENGTH)):
        # extract text
        chunk_text = sample["dialogue"][i : i + CHUNK_LENGTH]
        # create dictionary with the chunked text and metadata
        chunks.append(
            # remove uncompleted sentences with string split
            {
                "text": ".".join(chunk_text.split(".")[1:-1]).lstrip(),
                "metadata": {"page": c, "num_words": len(chunk_text)},
            }
        )
    # create new column
    sample["chunks"] = chunks
    return sample


def _explore_genres(df, genres):
    """
    Method that samples text snippets from a movie dialogue after filtering for specified genre.
    """
    snippets = {}
    for genre in genres:
        try:
            snippets[genre] = (
                df[df["genre"] == genre].sample(1).iloc[0]["dialogue"][2000:2500]
            )
        except:
            snippets[genre] = (
                df[df["genre"] == genre].sample(1).iloc[0]["dialogue"][:500]
            )
    output = ""
    for genre in genres:
        output = output + f"{genre}: , {snippets[genre]}\n\n"
    return print(output)


def _explore_df(df):
    """
    Method that samples text snippets from movie dialogues across multiple pre-defined genres.
    """
    try:
        crime_snippet = (
            df[df["genre"] == "action"].sample(1).iloc[0]["dialogue"][2000:2500]
        )
        comedy_snippet = (
            df[df["genre"] == "comedy"].sample(1).iloc[0]["dialogue"][2000:2500]
        )
    except:
        crime_snippet = df[df["genre"] == "action"].sample(1).iloc[0]["dialogue"][:500]
        comedy_snippet = df[df["genre"] == "comedy"].sample(1).iloc[0]["dialogue"][:500]

    return print(
        "Action: ..."
        + crime_snippet
        + "...\n\n"
        + "Comedy: ..."
        + comedy_snippet
        + "..."
    )


def _prepare_data():
    """
    Method to download Cornell movie corpus, extract the dialogue, movie name and genre to return a dataframe.
    """
    # download data
    corpus = Corpus(filename=download("movie-corpus", verbose=False))

    # obtain keys for all dialogs across various movies
    utter_keys = list(corpus.utterances.keys())
    convo_keys = list(corpus.conversations.keys())

    # initialize dataframe
    movie_df = pd.DataFrame(columns=["movie", "dialogue"])

    # create empty list to store movie name, dialogue
    movie_ls = []
    text_ls = []
    genre_dict = dict()

    # loop through all utterances and append to list
    for u in utter_keys:
        movie_ls.append(corpus.utterances[u].speaker.meta["movie_name"])
        text_ls.append(corpus.utterances[u].text)

    # loop through conversations and append to dictionary
    for c in convo_keys:
        try:
            genre_dict[corpus.conversations[c].meta["movie_name"]] = ast.literal_eval(
                corpus.conversations[c].meta["genre"]
            )[0]
        except:
            genre_dict[corpus.conversations[c].meta["movie_name"]] = "none"

    # fill dataframe with data
    movie_df["movie"] = movie_ls
    movie_df["dialogue"] = text_ls

    # group by movie title and concatenate all text into one long dialogue
    grouped_df = (
        movie_df.groupby("movie")["dialogue"].apply(lambda x: " ".join(x)).reset_index()
    )

    # join with genre data
    grouped_df = grouped_df.merge(
        pd.DataFrame({"movie": genre_dict.keys(), "genre": genre_dict.values()}),
        on="movie",
    )

    # delete objects that are no longer in use
    del corpus, utter_keys, convo_keys, movie_df, movie_ls, text_ls, genre_dict

    # garbage collect
    gc.collect()

    return grouped_df


def _return_prompt_and_responses(samples, batch_multiplier) -> Dict[str, str]:
    """
    Create correct format for DPO steps.
    """
    return {
        "prompt": [
            """Write a summary of this chunk of movie dialogue delimited by triple backquotes that includes the main points and any important details."""
        ]
        * batch_multiplier,
        "chosen": samples["summary"],  # rated better than k
        "rejected": samples["toxic_summary"],  # rated worse than j
    }


def _replace_nouns_with_list(text, replacement_probability=0.3):
    """
    Method to inject toxicity into text string with user defined probability by replacing nouns.
    """
    # open file from code package that contains profanities
    with open(
        os.path.dirname(better_profanity.__file__) + "/profanity_wordlist.txt", "r"
    ) as file:
        # read the file contents and store in list
        replacement_list = file.read().splitlines()

    # Process the text using spaCy
    doc = nlp(text)

    # Replace nouns with words from the replacement list
    replaced_text = []
    for token in doc:
        if token.pos_ == "NOUN" and random.random() < replacement_probability:
            replaced_text.append(
                random.choice(replacement_list) if replacement_list else token.text
            )
        else:
            replaced_text.append(token.text)

    # Join the words back into a string
    result = " ".join(replaced_text)

    return result


def _map_columns(sample):
    """
    Method to overwrite non-toxic summary with toxicity injected summaries for educational purpose.
    """
    if sample["genre"] in ["action", "crime"]:
        sample["summary"] = sample["toxic_summary"]
    else:
        sample["summary"] = sample["summary"]
    return sample
