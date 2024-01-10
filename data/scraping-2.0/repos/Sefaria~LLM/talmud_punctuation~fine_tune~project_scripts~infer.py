import csv
import django
django.setup()
import typer
import json
from sefaria.model import *
from sefaria.utils.hebrew import strip_cantillation
import random
import os
from langchain.chat_models import ChatOpenAI
import openai
import re
from sefaria.helper.normalization import NormalizerComposer, RegexNormalizer, AbstractNormalizer
from util.general import get_removal_list

api_key = os.getenv("OPENAI_API_KEY")


seed_value = 245
random.seed(seed_value)
model_name = "ft:gpt-3.5-turbo-0613:sefaria:he-punct:8ClpgehI"
system_message = "Punctuate this Talmudic passage based on the commentary I provide. Extract the relevant punctuation marks (, : . ? ! \\\"\\\" -) from the commentary and put them in the original. Output only the original Aramaic passage with punctuation without \\\"cantilation\\\" or \\\"nikud\\\".\\n"
def get_response_openai(original_passage, commentary):
    user_message = "Original Talmudic Passage:\n" + original_passage + '\n' + "Commentary:\n" + commentary
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_message
            }

        ],
        temperature=1,
        max_tokens=600,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # print(response)
    inference = response["choices"][0]["message"]["content"]
    return(inference)

def get_response_openai_try_again(original_passage, commentary, previous_inference):
    user_message = "Original Talmudic Passage:\n" + original_passage + '\n' + "Commentary:\n" + commentary
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_message
            },
            {
              "role": "assistant",
              "content": previous_inference
            },
            {
                "role": "user",
                "content": "continue"
            },

        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    inference = response["choices"][0]["message"]["content"]
    return(inference)

def read_csv(file_path):
    """
    Read a CSV file and return a list of tuples.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - list of tuples: Each tuple represents a row in the CSV file.
    """
    data = []
    with open(file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            data.append(tuple(row))
    return data
def write_tuples_to_csv(data, csv_filename):
    """
    Write a list of tuples to a CSV file.

    Parameters:
    - data: List of tuples to be written to CSV.
    - csv_filename: Name of the CSV file to be created or overwritten.
    """
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header if needed
        # csv_writer.writerow(['Column1', 'Column2', ...])  # Uncomment and replace with your column names

        # Write data
        csv_writer.writerows(data)

def find_first_occurrence_indices(text, regex_pattern):
    """
    Returns:
    - tuple: A tuple containing the start and end indexes of the first match, or None if no match is found.
    """
    match = re.search(regex_pattern, text)
    if match:
        return match.span()
    else:
        return None
def remove_first_occurrence(text, regex_pattern):
    return re.sub(regex_pattern, '', text, count=1)
def compute_old_indices_from_punctuated(punctuated_text):
    old_indices = []
    punctuationre = re.compile(
        r'[\.\!\?\:\,\u05F4](?![\u0591-\u05bd\u05bf-\u05c5\u05c7\u200d\u05d0-\u05eA](?:[\.\!\?\:\,\u05F4\s]|$))|—\s')
    while(True):
        span = find_first_occurrence_indices(punctuated_text, punctuationre)
        if span:
            old_indices.append((span[0], span[0]))
        else:
            break
        punctuated_text = remove_first_occurrence(punctuated_text, punctuationre)
    return (old_indices)
def find_all_tokens(regex_pattern, text):
    """
    Find all tokens captured by a regex pattern in the given text.

    Parameters:
    - regex_pattern (str): The regular expression pattern.
    - text (str): The input text.

    Returns:
    - list: A list of all tokens captured by the regex pattern.
    """
    matches = re.findall(regex_pattern, text)
    return matches

# def get_tuples_with_element(lst, i, x):
#     return [t for t in lst if len(t) > i and t[i] == x]
def insert_tokens_at_spans(s, spans, tokens):
    result = list(s)
    result.append('')
    starts = [t[0] for t in spans]

    for i, element in enumerate(result):
        if i in starts:
            result[i] = tokens[0] + result[i]
            tokens = tokens[1:]

    return ''.join(result)

def remove_tokens(regex_pattern, text):
    return re.sub(regex_pattern, '', text)

def realign_entities(punctuated_text: str, vocalised_text: str) -> str:
    punctuationre = re.compile(
        r'[\.\!\?\:\,\u05F4]+(?![\u0591-\u05bd\u05bf-\u05c5\u05c7\u200d\u05d0-\u05eA](?:[\.\!\?\:\,\u05F4\s]|$))|—\s')
    unpuncutated_text = remove_tokens(punctuationre, punctuated_text)

    removal_list = get_removal_list(vocalised_text, unpuncutated_text)
    temp_normalizer = AbstractNormalizer()
    mapping = temp_normalizer.get_mapping_after_normalization(vocalised_text, removal_list, reverse=False)
    old_inds = compute_old_indices_from_punctuated(punctuated_text)
    # a = insert_tokens_at_spans("Moses Is the best", compute_old_indices_from_punctuated("Moses. Is, the best!"), find_all_tokens(punctuationre, "Moses. Is, the best!"))
    a = insert_tokens_at_spans(unpuncutated_text, old_inds, find_all_tokens(punctuationre, punctuated_text))
    new_inds = temp_normalizer.convert_normalized_indices_to_unnormalized_indices(old_inds, mapping, reverse=False)
    vocalised_text = insert_tokens_at_spans(vocalised_text, new_inds, find_all_tokens(punctuationre, punctuated_text))

def punctuate_single_word(punctuated_word, unpunctuated_word):
    punctuations_end_one_char = {'.', ',', ';', ':',  '!', '?', "״"}
    punctuations_end_two_chars = {'?!'}
    punctuated_word_no_heifen = punctuated_word.replace('—', '')

    if len(punctuated_word_no_heifen) >= 4 and punctuated_word[-3] in punctuations_end_one_char and punctuated_word_no_heifen[-2] in punctuations_end_one_char and punctuated_word_no_heifen[-1] in punctuations_end_one_char:
        unpunctuated_word += punctuated_word_no_heifen[-3:]
    elif len(punctuated_word_no_heifen) >= 3 and punctuated_word_no_heifen[-2] in punctuations_end_one_char and punctuated_word_no_heifen[-1] in punctuations_end_one_char:
        unpunctuated_word += punctuated_word_no_heifen[-2:]
    elif len(punctuated_word_no_heifen) >= 2 and punctuated_word_no_heifen[-1] in punctuations_end_one_char:
        unpunctuated_word += punctuated_word_no_heifen[-1]


    if len(punctuated_word_no_heifen) >= 2 and punctuated_word_no_heifen[0] == "״":
        unpunctuated_word = "״" + unpunctuated_word

    if punctuated_word.endswith('—'):
        unpunctuated_word += ' —'

    return unpunctuated_word


def is_subsequence(sub, main):
    it = iter(main)
    return all(item in it for item in sub)
def punctuate_vocalised(punctuated_text: str, vocalised_text: str) -> str:
    if "ועשה על פיהם״, ״שוגג״ למה לי?" in punctuated_text:
        halt = True
    punctuated_text_list = punctuated_text.replace(' —', '—').split()
    vocalised_text_list = vocalised_text.split()
    vocalised_text_list_suffix = vocalised_text.split()
    # if len(punctuated_text_list) != len(vocalised_text_list):
    #     print("Oh!")
    punctuationre = re.compile(
        r'[\.\!\?\:\,\u05F4]+(?![\u0591-\u05bd\u05bf-\u05c5\u05c7\u200d\u05d0-\u05eA](?:[\.\!\?\:\,\u05F4\s]|$))|—\s')
    matches = []
    global_vocalized_index = 0
    for puncutated_word in punctuated_text_list:
        unpuncutated_word = puncutated_word.replace('—', '')
        unpuncutated_word = remove_tokens(punctuationre, unpuncutated_word)
        for index, vocalised_word in enumerate(vocalised_text_list_suffix):
            if is_subsequence(list(unpuncutated_word), list(vocalised_word)):
                vocalised_text_list_suffix = vocalised_text_list_suffix[index+1:]
                global_vocalized_index += index
                matches += [(puncutated_word, vocalised_word, global_vocalized_index)]
                vocalised_text_list[global_vocalized_index] = punctuate_single_word(puncutated_word, vocalised_word)
                global_vocalized_index += 1
                break

    return  ' '.join(vocalised_text_list)




if __name__ == '__main__':
    # typer.run(visualize)
    inferences = []
    inferences.append(("Ref", "Original", "Inference"))
    punctuationre = re.compile(
        r'[\.\!\?\:\,\u05F4]+(?![\u0591-\u05bd\u05bf-\u05c5\u05c7\u200d\u05d0-\u05eA](?:[\.\!\?\:\,\u05F4\s]|$))|—\s')

    # ref_text_pairs = [(seg.tref, seg.text('he', "William Davidson Edition - Aramaic").text,  Ref("Steinsaltz on " + seg.tref).text('he').text) for seg in Ref("Horayot").all_segment_refs()]
    #
    # for pair in ref_text_pairs:
    #     # print(pair[2])
    #     inference = get_response_openai(pair[1], pair[2])
    #     inferences.append((pair[0], pair[1], inference))
    #     # print(inference)
    #     no_punct = punctuationre.sub('', inference)
    #     if len(punctuationre.sub('', inference).split()) < len(pair[1].split()):
    #         print("omission!")
    #         print(pair[0])
    #         # print(get_response_openai_try_again(pair[1], pair[2], inference))
    # write_tuples_to_csv(inferences, "horayot_inferences.csv")

    tuples = read_csv("horayot_inferences.csv")[1:]
    ref_original_punctuated_vocalised = []
    for tuple in tuples:
        ref_original_punctuated_vocalised.append((tuple[0], tuple[1], tuple[2], (Ref(tuple[0]).text('he', "William Davidson Edition - Vocalized Aramaic").text)))

    ref_original_punctuated_vocalised_punctuatedvocalized = [("Ref", "Original", "Inference", "Original Vocalized", "Inference Vocalized")]
    for tuple in ref_original_punctuated_vocalised:
        # a = compute_old_indices_from_punctuated(tuple[1])
        punctuated_and_vocalized = punctuate_vocalised(tuple[2], tuple[3])
        ref_original_punctuated_vocalised_punctuatedvocalized += [(tuple[0], tuple[1], tuple[2], tuple[3], punctuated_and_vocalized)]
    write_tuples_to_csv(ref_original_punctuated_vocalised_punctuatedvocalized, "horayot_inferences_vocalized.csv")

