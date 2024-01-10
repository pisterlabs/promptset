import numpy as np
from sentence_transformers import SentenceTransformer, util

from openai.embeddings_utils import cosine_similarity, get_embedding

model = SentenceTransformer('all-MiniLM-L6-v2')

def openai_encode(sentences, model="text-embedding-ada-002"):
    """
    Encode sentences using OpenAI's ADA.
    :param sentences: list of sentences
    :return: list of embeddings
    """
    embeddings = []
    for sentence in sentences:
        sentence = sentence.replace("\n", " ")
        sentence = sentence.replace("\t", " ")
        sentence = sentence.replace("\r", " ")
        sentence = sentence.replace("  ", " ")
        sentence = sentence.strip()
        #embeddings.append(openai.embeddings.create(input = [sentence], model=model).data[0].embedding)
        embedding = np.zeros(1536).tolist()
        if len(sentence) > 0:
            embedding = get_embedding(sentence, engine=model)
        embeddings.append(embedding)
    return embeddings


def get_cosine_similarity_score(sentences1, sentences2):
    #Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)
    # embeddings1 = openai_encode(sentences1)
    # embeddings2 = openai_encode(sentences2)

    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    return cosine_scores

def get_cosine_similarity_scores(sentences1, sentences2):
    if (len(sentences1) == 0 or len(sentences2) == 0):
        return [], []
    #Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)
    # embeddings1 = openai_encode(sentences1)
    # embeddings2 = openai_encode(sentences2)

    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    pairs = []
    for i in range(len(sentences1)):
        for j in range(len(sentences2)):
            pairs.append({'index': [i, j], 'score': cosine_scores[i][j].item()})

    #Sort scores in decreasing order
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

    top_10_pairs = []

    for pair in pairs[0:10]:
        i, j = pair['index']
        top_10_pairs.append({
            "sentence1": sentences1[i],
            "sentence2": sentences2[j],
            "score": pair['score']
        })

    return cosine_scores.tolist(), top_10_pairs

def test():
    # Two lists of sentences
    sentences1 = ['The cat sits outside',
                'A man is playing guitar',
                'The new movie is awesome']

    sentences2 = ['The dog plays in the garden',
                'A woman watches TV',
                'The new movie is so great']

    #Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)
    # embeddings1 = openai_encode(sentences1)
    # embeddings2 = openai_encode(sentences2)

    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    
    #Output the pairs with their score
    for i in range(len(sentences1)):
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))

def main():
    test()

if __name__ == "__main__":
    main()