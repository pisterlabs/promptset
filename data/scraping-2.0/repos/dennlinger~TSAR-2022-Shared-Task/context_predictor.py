"""
Uses a long context prediction setting for GPT-3.
"""

import os
import json
import regex
from collections import defaultdict
from typing import List, Tuple, Dict

from tqdm import tqdm
import openai

from config import API_KEY


def clean_predictions(text: str, given_word: str) -> List[str]:
    """
    Post-processing of files, by trying different strategies to coerce it into actual singular predictions.
    :param text: Unfiltered text predicted by a language model
    :param given_word: The word that is supposed to be replaced. Sometimes appears in `text`.
    :return: List of individual predictions
    """

    # Catch sample 248
    if text.startswith(given_word):
        text = text[len(given_word):]

    # Clear additional clutter that might have been encountered
    text = text.strip("\n :;.?!")

    # Presence of newlines within the prediction indicates prediction as list
    if "\n" in text.strip("\n "):
        cleaned_predictions = text.strip("\n ").split("\n")

    # Other common format contained comma-separated list without anything else
    elif "," in text.strip("\n "):
        cleaned_predictions = [pred.strip(" ") for pred in text.strip("\n ").split(",")]

    # Sometimes in-line enumerations also occur, this is a quick check to more or less guarantee
    # at least 6 enumerated predictions
    elif "1." in text and "6." in text:
        cleaned_predictions = regex.split(r"[0-9]{1,2}\.?", text.strip("\n "))

    else:
        raise ValueError(f"Unrecognized list format in prediction '{text}'")

    # Edge case where there is inconsistent newlines
    if 2 < len(cleaned_predictions) < 5:
        raise ValueError(f"Inconsistent newline pattern found in prediction '{text}'")

    # Remove numerals
    cleaned_predictions = [remove_numerals(pred) for pred in cleaned_predictions]
    # Make sure everything is lower-cased and stripped
    cleaned_predictions = [pred.lower().strip(" \n") for pred in cleaned_predictions]
    # Remove "to" in the beginning
    cleaned_predictions = [remove_to(pred) for pred in cleaned_predictions]
    # Remove predictions that match the given word
    cleaned_predictions = remove_identity_predictions(cleaned_predictions, given_word)
    # Remove empty predictions that may have slipped through:
    cleaned_predictions = remove_empty_predictions(cleaned_predictions)
    # Remove multi-word predictions (with 3 or more steps)
    cleaned_predictions = remove_multiwords(cleaned_predictions)
    # Remove punctuation
    cleaned_predictions = remove_punctuation(cleaned_predictions)

    return cleaned_predictions


def remove_punctuation(predictions: List[str]) -> List[str]:
    return [prediction.strip(".,;?!") for prediction in predictions]


def remove_multiwords(predictions: List[str], max_segments: int = 2) -> List[str]:
    return [prediction for prediction in predictions if len(prediction.split(" ")) <= max_segments]


def remove_empty_predictions(predictions: List[str]) -> List[str]:
    return [pred for pred in predictions if pred.strip("\n ")]


def remove_identity_predictions(predictions: List[str], given_word: str) -> List[str]:
    return [pred for pred in predictions if pred != given_word]


def remove_numerals(text: str) -> str:
    """
    Will remove any leading numerals (optionally with a dot).
    :param text: Input text, potentially containing a leading numeral
    :return: cleaned text
    """

    return regex.sub(r"[0-9]{1,2}\.? ?", "", text)


def remove_to(text: str) -> str:
    """
    Removes the leading "to"-infinitive from a prediction, which is sometimes caused when the context word
    is preceeded with a "to" in the text.
    :param text: Prediction text
    :return: Text where a leading "to " would be removed from the string.
    """
    return regex.sub(r"^to ", "", text)


def deduplicate_predictions(predictions: List[Tuple]) -> Dict:
    """
    Slightly less efficient deduplication method that preserves "ranking order" by appearance.
    :param predictions: List of predictions
    :return: Filtered list of predictions that no longer contains duplicates.
    """
    merged = defaultdict(float)
    for prediction, score in predictions:
        merged[prediction] += score

    return merged


def get_highest_predictions(predictions: Dict, number_predictions: int) -> List[str]:
    return [prediction for prediction, _ in sorted(predictions.items(), key=lambda item: item[1], reverse=True)][:number_predictions]


def assign_prediction_scores(predictions: List[str], start_weight: float = 5.0, decrease: float = 0.5) -> List[Tuple]:
    """
    The result of   predictions - len(predictions) * decrease   should equal 0.
    :param predictions:
    :param start_weight:
    :param decrease:
    :return:
    """
    weighted_predictions = []
    for idx, prediction in enumerate(predictions):
        weighted_predictions.append((prediction, start_weight - idx * decrease))

    return weighted_predictions


def get_prompts_and_temperatures(context: str, word: str) -> List[Tuple[str, str, float]]:

    zero_shot = f"Context: {context}\n" \
                f"Question: Given the above context, list ten alternative words for \"{word}\" that are easier to understand.\n" \
                f"Answer:"

    no_context_zero_shot = f"Give me ten simplified synonyms for the following word: {word}"

    no_context_single_shot = f"Question: Find ten easier words for \"compulsory\".\n" \
                             f"Answer:\n" \
                             f"1. mandatory\n2. required\n3. essential\n4. forced\n5. important\n" \
                             f"6. necessary\n7. obligatory\n8. unavoidable\n9. binding\n10. prescribed\n\n" \
                             f"Question: Find ten easier words for \"{word}\".\n" \
                             f"Answer:"

    single_shot_prompt = f"Context: A local witness said a separate group of attackers disguised in burqas — the head-to-toe robes worn by conservative Afghan women — then tried to storm the compound.\n" \
                         f"Question: Given the above context, list ten alternative words for \"disguised\" that are easier to understand.\n" \
                         f"Answer:\n" \
                         f"1. concealed\n2. dressed\n3. hidden\n4. camouflaged\n5. changed\n" \
                         f"6. covered\n7. masked\n8. unrecognizable\n9. converted\n10. impersonated\n\n"\
                         f"Context: {context}\n" \
                         f"Question: Given the above context, list ten alternative words for \"{word}\" that are easier to understand.\n" \
                         f"Answer:"

    few_shot_prompt = f"Context: That prompted the military to deploy its largest warship, the BRP Gregorio del Pilar, which was recently acquired from the United States.\n" \
                      f"Question: Given the above context, list ten alternative words for \"deploy\" that are easier to understand.\n" \
                      f"Answer:\n" \
                      f"1. send\n2. post\n3. use\n4. position\n5. send out\n" \
                      f"6. employ\n7. extend\n8. launch\n9. let loose\n10. organize\n\n" \
                      f"Context: The daily death toll in Syria has declined as the number of observers has risen, but few experts expect the U.N. plan to succeed in its entirety.\n" \
                      f"Question: Given the above context, list ten alternative words for \"observers\" that are easier to understand.\n" \
                      f"Answer:\n" \
                      f"1. watchers\n2. spectators\n3. audience\n4. viewers\n5. witnesses\n" \
                      f"6. patrons\n7. followers\n8. detectives\n9. reporters\n10. onlookers\n\n" \
                      f"Context: {context}\n" \
                      f"Question: Given the above context, list ten alternative words for \"{word}\" that are easier to understand.\n" \
                      f"Answer:"

    # Mix between different methods
    prompts = [("conservative zero-shot with context", zero_shot, 0.3),
               ("creative zero-shot with context", zero_shot, 0.8),
               ("zero-shot without context", no_context_zero_shot, 0.7),
               ("single-shot without context", no_context_single_shot, 0.6),
               ("single-shot with context", single_shot_prompt, 0.5),
               ("few-shot with context", few_shot_prompt, 0.5)]

    return prompts


if __name__ == '__main__':
    debug = False
    max_number_predictions = 10
    continue_from = 0

    if debug:
        with open("datasets/trial/tsar2022_en_trial_none.tsv") as f:
            lines = f.readlines()
    else:
        with open("datasets/test/tsar2022_en_test_none.tsv") as f:
            lines = f.readlines()

    openai.api_key = API_KEY

    baseline_predictions = []
    ensemble_predictions = []

    if debug:
        lines = lines[:2]

    for idx, line in enumerate(tqdm(lines)):
        # Skip already processed samples
        if idx < continue_from:
            continue

        aggregated_predictions = []

        # Extract context and complex word
        context, word = line.strip("\n ").split("\t")

        # Get "ensemble prompts"
        prompts_and_temps = get_prompts_and_temperatures(context, word)

        for prompt_name, prompt, temperature in tqdm(prompts_and_temps):
            # Have not experimented too much with other parameters, but these generally worked well.
            response = openai.Completion.create(
                model="text-davinci-002",
                prompt=prompt,
                stream=False,
                temperature=temperature,
                max_tokens=256,
                top_p=1,
                best_of=1,
                frequency_penalty=0.5,
                presence_penalty=0.3
            )
            predictions = response["choices"][0]["text"]

            predictions = clean_predictions(predictions, word)
            weighted_predictions = assign_prediction_scores(predictions)
            aggregated_predictions.extend(weighted_predictions)

            # Store the "conservative zero-shot with context" predictions as a baseline run.
            if prompt_name == "conservative zero-shot with context":
                baseline_predictions.append(weighted_predictions)
                with open("tsar2022_test_en_UniHD_1.tsv", "a") as f:
                    prediction_string = "\t".join(predictions[:max_number_predictions])
                    f.write(f"{context}\t{word}\t{prediction_string}\n")

        aggregated_predictions = deduplicate_predictions(aggregated_predictions)
        highest_scoring_predictions = get_highest_predictions(aggregated_predictions, max_number_predictions)
        with open("tsar2022_test_en_UniHD_3.tsv", "a") as f:
            prediction_string = "\t".join(highest_scoring_predictions[:max_number_predictions])
            f.write(f"{context}\t{word}\t{prediction_string}\n")

        ensemble_predictions.append(aggregated_predictions)

        if debug:
            print(f"Complex word: {word}")
            print(f"{aggregated_predictions}")
            # break

    # FIXME: This currently overwrites previously generated scores!!!
    with open("baseline_scores.json", "w") as f:
        json.dump(baseline_predictions, f, ensure_ascii=False, indent=2)
    with open("ensemble_scores.json", "w") as f:
        json.dump(ensemble_predictions, f, ensure_ascii=False, indent=2)
