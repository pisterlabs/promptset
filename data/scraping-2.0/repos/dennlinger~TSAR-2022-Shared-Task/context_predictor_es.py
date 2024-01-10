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


def remove_multiwords(predictions: List[str], max_segments: int = 3) -> List[str]:
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
                f"Question: Given the above context, list ten alternative Spanish words for \"{word}\" that are easier to understand.\n" \
                f"Answer:"

    no_context_zero_shot = f"Give me ten simplified Spanish synonyms for the following word: {word}"

    no_context_single_shot = f"Question: Find ten easier Spanish words for \"folclório\".\n" \
                             f"Answer:\n" \
                             f"1. popular\n2. tradicional\n3. local\n4. de folclore\n5. de musica folk\n" \
                             f"6. costumbrista\n7. pintoresco\n8. típico\n9. de folclor\n10. típico\n" \
                             f"Question: Find ten easier Spanish words for \"{word}\".\n" \
                             f"Answer:"

    single_shot_prompt = f"Context: Además de partidos de fútbol americano, el estadio ha sido utilizado para una " \
                         f"gran variedad de eventos, entre los que se destacan varios partidos de la selección " \
                         f"nacional de fútbol de los Estados Unidos, y fue el hogar del ahora difunto club de la MLS, " \
                         f"el Tampa Bay Mutiny.\n" \
                         f"Question: Given the above context, list ten alternative Spanish words for \"difunto\" that are easier to understand.\n" \
                         f"Answer:\n" \
                         f"1. muerto\n2. fallecido\n3. extinto\n4. inexistente\n5. finado\n6. desaparecido\n" \
                         f"7. acabado\n8. inactivo\n9. tieso\n10. moribundo\n\n"\
                         f"Context: {context}\n" \
                         f"Question: Given the above context, list ten alternative Spanish words for \"{word}\" that are easier to understand.\n" \
                         f"Answer:"

    few_shot_prompt = f"Context: El texto denominado \"Lamentos de Ipuwer\" describe una situación caótica: reyes " \
                      f"desacreditados, invasión asiática del Delta, desórdenes revolucionarios, destrucción de " \
                      f"archivos y tumbas reales, ateísmo y divulgación de secretos religiosos.\n" \
                      f"Question: Given the above context, list ten alternative Spanish words for \"desacreditados\" that are easier to understand.\n" \
                      f"Answer:\n" \
                      f"1. difamados\n2. desprestigiados\n3. malos\n4. desprestigiado\n5. afrentados\n" \
                      f"6. demeritados\n7. desmentidos\n8. sin prestigio\n9. olvidados\n10. denigrados\n\n" \
                      f"Context: Sufrió una importante reducción en su capacidad para poder acogerse a las normas de la FIFA para los estadios de fútbol.\n" \
                      f"Question: Given the above context, list ten alternative Spanish words for \"acogerse\" that are easier to understand.\n" \
                      f"Answer:\n" \
                      f"1. adaptarse\n2. sumarse\n3. incorporarse\n4. obedecer\n5. apegarse\n6. ampararse\n" \
                      f"7. aceptar\n8. asimilarse\n9. aplicarse\n10. aceptarse\n\n" \
                      f"Context: {context}\n" \
                      f"Question: Given the above context, list ten alternative Spanish words for \"{word}\" that are easier to understand.\n" \
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
        with open("datasets/trial/tsar2022_es_trial_none.tsv") as f:
            lines = f.readlines()
    else:
        with open("datasets/test/tsar2022_es_test_none.tsv") as f:
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
                with open("tsar2022_test_es_UniHD_1.tsv", "a") as f:
                    prediction_string = "\t".join(predictions[:max_number_predictions])
                    f.write(f"{context}\t{word}\t{prediction_string}\n")

        aggregated_predictions = deduplicate_predictions(aggregated_predictions)
        highest_scoring_predictions = get_highest_predictions(aggregated_predictions, max_number_predictions)
        with open("tsar2022_test_es_UniHD_3.tsv", "a") as f:
            prediction_string = "\t".join(highest_scoring_predictions[:max_number_predictions])
            f.write(f"{context}\t{word}\t{prediction_string}\n")

        ensemble_predictions.append(aggregated_predictions)

        if debug:
            print(f"Complex word: {word}")
            print(f"{aggregated_predictions}")
            # break

    # FIXME: This currently overwrites previously generated scores!!!
    with open("baseline_scores_es.json", "w") as f:
        json.dump(baseline_predictions, f, ensure_ascii=False, indent=2)
    with open("ensemble_scores_es.json", "w") as f:
        json.dump(ensemble_predictions, f, ensure_ascii=False, indent=2)
