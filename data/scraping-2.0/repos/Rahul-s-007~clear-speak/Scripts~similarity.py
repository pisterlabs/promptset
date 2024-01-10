import openai
import numpy as np

def similarity_score(script_txt,speech_txt):
    resp = openai.Embedding.create(
        input=[script_txt,speech_txt],
        engine="text-similarity-davinci-001")

    embedding_a = resp['data'][0]['embedding']
    embedding_b = resp['data'][1]['embedding']

    similarity_score = np.dot(embedding_a, embedding_b)
    print(similarity_score)
    score = (100 + (100*similarity_score))//2
    print(score)
    return score
