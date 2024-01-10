import time
import os
import random
from pathlib import Path
from typing import List, Optional, Generator, Union

from spacy.tokens import Doc, Span
import argparse

from data import load_gold_triplets
import spacy
from extract_examples import extract_examples
from conspiracies.docprocessing.relationextraction import (
    MarkdownPromptTemplate2,
    PromptTemplate,
)
import openai

# Conspiracies

from extract_utils import write_txt, ndjson_gen
from src.concat_split_contexts import (
    concat_context,
    tweet_from_context_text,
)


def build_coref_pipeline():
    nlp_coref = spacy.blank("da")
    nlp_coref.add_pipe("sentencizer")
    nlp_coref.add_pipe("allennlp_coref")

    return nlp_coref


def build_headword_extraction_pipeline():
    nlp = spacy.load("da_core_news_sm")
    nlp.add_pipe("sentencizer")
    nlp.add_pipe(
        "heads_extraction",
        config={"normalize_to_entity": True, "normalize_to_noun_chunk": True},
    )
    return nlp


def batch_generator(generator: Generator, n: int):
    """Batches a generator into batches of size n.

    Args:
        generator (Generator): Generator to batch
        n (int): Size of batches

    Yields:
        batch (list): List of elements from generator
    """
    batch = []
    for element in generator:
        batch.append(element)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def prepare_template(template: PromptTemplate, n_examples=9, cv=1):
    """Prepares a template by extracting examples from gold docs Uses the
    targets as examples since they are criteria balanced.

    Args:
        template (PromptTemplate): Template to prepare
        n_examples (int, optional): Number of examples to use for prompting.
            Defaults to 10.
        cv (int, optional): Number of times to do the extraction.
            Defaults to 1, since this is only relevant for prompt comparing

    Returns:
        PromptTemplate: the template with examples
    """
    gold_docs = load_gold_triplets(nlp=spacy.load("da_core_news_sm"))
    targets, _ = extract_examples(gold_docs, n_examples, cv)
    return template(examples=targets[0])


def concat_resolve_unconcat_contexts(file_path: str):
    """Concatenates and resolves coreferences in contexts. The resolved
    contexts are then split, and the last (target) tweet is returned.

    Args:
        file_path (str): path to where the contexts are stored

    Returns:
        Generator: a generator with the resolved target tweets
    """
    context_tweets: List[str] = []
    for context in ndjson_gen(file_path):
        concatenated = concat_context(context)
        context_tweets.append(concatenated)

    coref_nlp = build_coref_pipeline()
    coref_docs = coref_nlp.pipe(context_tweets)
    resolved_docs = (d._.resolve_coref for d in coref_docs)

    resolved_tweets = (tweet_from_context_text(tweet) for tweet in resolved_docs)
    return resolved_tweets


def concatenate_tweets(file_path: str, save_path: Optional[str] = None):
    """Concatenates tweets in a file.

    Args:
        file_path (str): path to where the contexts are stored

    Returns:
        Generator: a generator with the concatenated tweets
    """
    context_tweets: List[str] = []
    for context in ndjson_gen(file_path):
        concatenated = concat_context(context)
        context_tweets.append(concatenated)
        if save_path:
            write_txt(save_path, [context[-1]["id"]], "a+")

    return context_tweets


def extract_save_triplets_gpt(
    responses: dict,
    template: PromptTemplate,
    event: str,
    nlp: spacy.Language,
):
    subjects, predicates, objects, triplets = [], [], [], []
    subjects_full, predicates_full, objects_full, triplets_full = [], [], [], []
    for response in responses["choices"]:
        for triplet in template.parse_prompt(response["text"], target_tweet=""):
            if "" in (triplet.subject, triplet.predicate, triplet.object):
                continue
            # Extract elements
            subject = nlp(triplet.subject)
            predicate = nlp(triplet.predicate)
            obj = nlp(triplet.object)

            # Save non-headword reduced
            subjects_full.append(triplet.subject)
            predicates_full.append(triplet.predicate)
            objects_full.append(triplet.object)
            triplets_full.append(
                f"{triplet.subject}, {triplet.predicate}, {triplet.object}",
            )

            # Save headword reduced
            subjects.append(subject._.most_common_ancestor.text)
            predicates.append(predicate._.most_common_ancestor.text)
            objects.append(obj._.most_common_ancestor.text)
            triplets.append(
                f"{subject._.most_common_ancestor.text}, {predicate._.most_common_ancestor.text}, {obj._.most_common_ancestor.text}",
            )

    write_txt(
        os.path.join("extracted_triplets_tweets", event, "subjects.txt"),
        subjects,
        "a+",
    )
    write_txt(
        os.path.join("extracted_triplets_tweets", event, "predicates.txt"),
        predicates,
        "a+",
    )
    write_txt(
        os.path.join("extracted_triplets_tweets", event, "objects.txt"),
        objects,
        "a+",
    )
    write_txt(
        os.path.join("extracted_triplets_tweets", event, "triplets.txt"),
        triplets,
        "a+",
    )
    write_txt(
        os.path.join("extracted_triplets_tweets", event, "subjects_full.txt"),
        subjects_full,
        "a+",
    )
    write_txt(
        os.path.join("extracted_triplets_tweets", event, "predicates_full.txt"),
        predicates_full,
        "a+",
    )
    write_txt(
        os.path.join("extracted_triplets_tweets", event, "objects_full.txt"),
        objects_full,
        "a+",
    )
    write_txt(
        os.path.join("extracted_triplets_tweets", event, "triplets_full.txt"),
        triplets_full,
        "a+",
    )


def prompt_gpt3(
    concatenated_tweets: Generator,
    api_key: Union[str, None],
    event: str,
    batch_size: int = 20,
    org_id: Optional[str] = None,
):
    """Prompts GPT-3 with a template and a generator of concatenated tweets.
    The tweets are resolved with coreference resolution, and then the template
    is used to create a prompt for each tweet. The prompts are then sent to
    GPT-3, and the responses are parsed and saved.

    Args:
        template (PromptTemplate): Template to use for prompting
        concatenated_tweets (Generator): Generator of concatenated tweets
        api_key (str): API key for GPT-3
        event (str): Event name
        batch_size (int, optional): Number of tweets to send in each batch.
            Defaults to 20.
        org_id (Optional[str], optional): Organization ID for GPT-3. Defaults to None.
    """
    assert api_key, "Please provide an API key when using GPT-3"

    # Setup
    print("Loading gold docs and setting up template")
    template = prepare_template(MarkdownPromptTemplate2)
    coref_nlp = build_coref_pipeline()
    head_nlp = build_headword_extraction_pipeline()
    if org_id:
        openai.org_id = org_id
    openai.api_key = api_key
    # Check if the folder for the event exists, if not create it
    Path(os.path.join("extracted_triplets_tweets", event)).mkdir(
        parents=True,
        exist_ok=True,
    )

    print("batching")
    for i, batch in enumerate(batch_generator(concatenated_tweets, batch_size)):
        start = time.time()
        coref_docs = coref_nlp.pipe(batch)
        resolved_docs = (d._.resolve_coref for d in coref_docs)
        resolved_target_tweets = (
            tweet_from_context_text(tweet) for tweet in resolved_docs
        )
        prompts = [
            template.create_prompt(target=tweet) for tweet in resolved_target_tweets
        ]

        print(f"sending request for batch {i}")
        while True:
            try:
                responses = openai.Completion.create(
                    model="text-davinci-002",
                    prompt=prompts,
                    temperature=0.7,
                    max_tokens=500,
                )

                print("parsing response and saving")
                extract_save_triplets_gpt(responses, template, event, head_nlp)
                print(f"batch {i} done in {time.time() - start} seconds\n")
                break

            except openai.error.InvalidRequestError as e:
                print("Invalid request got error: ", e)
                print("Retrying with fewer examples...")
                # Randomly select an example to drop
                current_examples: List[Doc] = list(template.examples)
                current_examples.pop(random.randrange(len(current_examples)))
                template.set_examples(current_examples)  # type: ignore
                prompts = [
                    template.create_prompt(target=tweet)
                    for tweet in resolved_target_tweets
                ]

            except openai.error.APIConnectionError:
                print("Connection reset, waiting 20 sec then retrying...")
                time.sleep(20)
            except openai.error.APIError:
                print("API error, waiting 20 sec then retrying...")
                time.sleep(20)
                continue
            except openai.error.RateLimitError:
                print("RateLimitError, waiting 20 sec then retrying...")
                time.sleep(20)
                continue


def multi2oie_extraction(
    concatenated_tweets: list,
    event: str,
):
    # Check if the folder for the event exists, if not create it
    Path(os.path.join("extracted_triplets_tweets", f"{event}_multi")).mkdir(
        parents=True,
        exist_ok=True,
    )
    print("Building pipeline")
    nlp = spacy.load("da_core_news_sm")
    nlp.add_pipe("sentencizer")
    nlp.add_pipe(
        "heads_extraction",
        config={"normalize_to_entity": True, "normalize_to_noun_chunk": True},
    )
    config = {"confidence_threshold": 2.7, "model_args": {"batch_size": 10}}
    nlp.add_pipe("relation_extractor", config=config)

    docs = nlp.pipe(concatenated_tweets)
    start = time.time()
    i = 0
    run = True
    while run:
        try:
            i += 1
            doc = next(docs)
        except KeyError:
            print("Received a KeyError, skipping tweet")
            docs = nlp.pipe(concatenated_tweets[i:])
            continue
        except StopIteration:
            print("Stopping iteration because of StopIteration exception")
            # TODO: the last iteration happens twice with this logic
            run = False
        subjects, predicates, objects, triplets = [], [], [], []
        for triplet in doc._.relation_triplets:
            if not all(isinstance(element, Span) for element in triplet):
                continue
            if len(triplet) != 3:
                continue
            subject = triplet[0]._.most_common_ancestor.text
            predicate = triplet[1]._.most_common_ancestor.text
            obj = triplet[2]._.most_common_ancestor.text
            subjects.append(subject)
            predicates.append(predicate)
            objects.append(obj)
            triplets.append((subject, predicate, obj))

        if len(triplets) > 0:
            write_txt(
                os.path.join(
                    "extracted_triplets_tweets",
                    f"{event}_multi",
                    "subjects.txt",
                ),
                subjects,
                "a+",
            )
            write_txt(
                os.path.join(
                    "extracted_triplets_tweets",
                    f"{event}_multi",
                    "predicates.txt",
                ),
                predicates,
                "a+",
            )
            write_txt(
                os.path.join(
                    "extracted_triplets_tweets",
                    f"{event}_multi",
                    "objects.txt",
                ),
                objects,
                "a+",
            )
            write_txt(
                os.path.join(
                    "extracted_triplets_tweets",
                    f"{event}_multi",
                    "triplets.txt",
                ),
                triplets,
                "a+",
            )
        if i % 20 == 0 and i != 0:
            print(
                f"Tweet {i} done. Processed 20 tweets in {time.time() - start} seconds\n",
            )
            start = time.time()


def main(
    file_path: str,
    event: str,
    api_key: Optional[str] = None,
    template_batch_size: int = 20,
    sample_size: Union[None, int] = None,
    org_id: Optional[str] = None,
    extraction_method: str = "gpt3",
):
    """Main function for extracting triplets from tweets Uses GPT3 prompting.

    Args:
        file_path (str): path to where the contexts are stored
        event (str): name of the event. Used when saving the extracted triplets
        template_batch_size (int, optional): Number of tweets to prompt GPT3 with at a time.
            Defaults to 20 (max length for list of prompts).
    """
    print("Concatenating tweets")
    concatenated = concatenate_tweets(file_path, f"processed_tweets_{file_path}.txt")

    # Downsampling
    if sample_size:
        if len(concatenated) < sample_size:
            print(
                f"Sample size ({sample_size}) is larger than the number of tweets ({len(concatenated)}), using all tweets",
            )
        else:
            concatenated = random.sample(concatenated, sample_size)
            print(f"Downsampled to {len(concatenated)} tweets")

    print(f"Extracting triplets with {extraction_method}")
    if extraction_method == "gpt3":
        prompt_gpt3(
            concatenated,
            api_key,
            event,
            template_batch_size,
            org_id,
        )
    elif extraction_method == "Multi2OIE":
        multi2oie_extraction(
            concatenated,
            event,
        )
    else:
        raise ValueError(
            f"Extraction method {extraction_method} not supported. Please choose between 'gpt3' and 'Multi2OIE'.",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--file_path",
        type=str,
        help="Path to the file containing the context tweets",
    )
    parser.add_argument(
        "-e",
        "--event",
        type=str,
        help="Name of the event. Used when saving the extracted triplets",
    )
    parser.add_argument(
        "-t",
        "--template_batch_size",
        type=int,
        default=20,
        required=False,
        help="Number of tweets to prompt GPT3 with at a time. Defaults to 20 (max length for list of prompts).",
    )
    parser.add_argument(
        "-s",
        "--sample_size",
        type=int,
        default=30000,
        required=False,
        help="Number of tweets to downsample dataset to. Defaults to 30000.",
    )
    parser.add_argument(
        "-o",
        "--org_id",
        type=str,
        required=False,
        default=None,
        help="OpenAI organization ID. Defaults to None.",
    )
    parser.add_argument(
        "-api_key",
        "--api_key",
        type=str,
        required=False,
        default=None,
        help="OpenAI API key. Defaults to None.",
    )
    parser.add_argument(
        "-x",
        "--extraction_method",
        type=str,
        required=False,
        default="gpt3",
        help="Extraction method. Defaults to gpt3. Other option is 'Multi2OIE'",
    )
    # file_path = os.path.join("src", "TESTnew_tweet_threads_2019-03-10_2019-03-17.ndjson")
    # event="covid_week_1"
    # template_batch_size=20
    args = parser.parse_args()
    main(
        file_path=args.file_path,
        event=args.event,
        api_key=args.api_key,
        template_batch_size=args.template_batch_size,
        sample_size=args.sample_size,
        org_id=args.org_id,
        extraction_method=args.extraction_method,
    )
