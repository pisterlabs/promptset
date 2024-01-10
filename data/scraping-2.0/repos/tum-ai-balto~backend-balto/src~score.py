from bleurt import score
import openai
import numpy as np
import re, os

CHECKPOINT = "bleurt/test_checkpoint"

openai.api_key = os.getenv('OPENAI_KEY')


def get_sentences(text: str):
    text = text.replace("\n", "")
    sentences = re.split('\:|\.|\?|\!|\¿|\¡', text)
    sentences[:] = [x for x in sentences if x]
    return sentences


def get_embedding(sentence):
    completion = openai.Embedding.create(model="text-embedding-ada-002", input=sentence)
    return completion["data"][0]["embedding"]


# Define a function to find the nearest list
def find_nearest_list(list1_item, list2):
    # Compute the differences between the given list and all the lists in list2
    differences = [np.abs(np.array(list1_item) - np.array(list2_item)) for list2_item in list2]
    # Find the index of the list in list2 with the smallest difference
    nearest_list_index = np.argmin([np.sum(difference) for difference in differences])
    # Return the nearest list
    return nearest_list_index


def keep_just_nearest_elements(embed1, embed2):
    if len(embed2) > len(embed1):
        # Put together each list of the first list with its nearest list of the second list
        result = [[index, find_nearest_list(embed1[index], embed2)] for index in range(len(embed1))]
    else:
        result = [[find_nearest_list(embed2[index], embed1), index] for index in range(len(embed2))]
    return result


def create_clean_list(nearest, list1, list2):
    clean1 = []
    clean2 = []
    for elem in nearest:
        clean1.append(list1[elem[0]])
        clean2.append(list2[elem[1]])
    return clean1, clean2


def calculate_score(clean1, clean2):
    scorer = score.BleurtScorer(CHECKPOINT)
    scores = np.array(scorer.score(references=clean1, candidates=clean2))
    return np.clip(scores, 0, 1), np.mean(scores[(scores > 0) & (scores < 1)])


def calculate_fidelity(text1, text2):
    sentences1 = get_sentences(text1)
    sentences2 = get_sentences(text2)

    print(len(sentences1), len(sentences2))
    print(sentences1)
    print(sentences2)

    embedding1 = list()
    embedding2 = list()

    for sentence in sentences1:
        embedding1.append(get_embedding(sentence))

    for sentence in sentences2:
        embedding2.append(get_embedding(sentence))

    # Print the result
    nearest = keep_just_nearest_elements(embedding1, embedding2)
    print(nearest)
    clean1, clean2 = create_clean_list(nearest, sentences1, sentences2)
    print(clean1)
    print(clean2)

    return calculate_score(clean1, clean2)
