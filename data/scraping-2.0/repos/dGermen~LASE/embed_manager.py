import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sklearn
import time

class EmbedManager:

    def __init__(self, 
                embed_data,
                openai_apikey = "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",
                model = "text-embedding-ada-002",
                 ):
        pass

        self.embed_data = embed_data
        openai.api_key = openai_apikey
        self.model = model

    def embed(self, text):

        done = False

        while not done:
        # Embed the text
            try:
                response = openai.Embedding.create(
                input=text,
                model=self.model
                )
                done = True

            except Exception as e:
                print(e)
                print("Rate limit error, waiting for 0.1 seconds")
                time.sleep(0.1)
                continue
            
        # Extract the embeddings
        embeddings = response['data'][0]['embedding']

        # List to numpy array
        embeddings = np.array(embeddings)
        
        return embeddings

    def embed_weighting(self, query_embed, ids, weights = 1):
        # Gets ids of the docs, finds their embeddings
        # Find representative embedding by weighted averaging

        if weights == 1:
            # If weights are not given, use equal weights
            weights = np.ones(len(ids))
        
        # Set query weight to the average of the weights
        q_weight = np.sum(weights) / 2

        # Append the query weight to the weights
        weights = np.append(weights, q_weight)

        # Scale weights
        weights = weights / np.sum(weights)

        
        # Get embeddings, store in 2d numpy array
        embeddings = np.array([self.find_embedding(id)[0][1:] for id in ids])
        
        # Append the query embedding to the embeddings
        embeddings = np.append(embeddings, [query_embed], axis=0)

        # Get weighted average
        weighted_average = np.average(embeddings, axis=0, weights=weights)

        return weighted_average

        

    def find_embedding(self, id):
        return self.embed_data[self.embed_data[:, 0] == id]
        pass

    def embeddings_knn(self, embedding, k):
        # Return the k nearest neighbor as IDs using embedding and self.embed_data

        # Compute cosine similarity between the random embedding and all embeddings in EMD
        similarities = cosine_similarity(self.embed_data[:, 1:], [embedding])
        min = similarities.min()
        max = similarities.max()

        # Sort the cosine similarity scores in descending order
        sorted_indices = np.argsort(similarities, axis=0)[::-1]

        # Retrieve the IDs of the top k most similar embeddings
        top_k_ids = self.embed_data[sorted_indices[:k], 0]

        # Retrieve similarity scores of the top k most similar embeddings
        top_k_scores = similarities[sorted_indices[:k], 0]

        # Scale the similarity scores to be between 0 and 1, use variables min max
        top_k_scores_scaled = (top_k_scores - min) / (max - min)

        # Create a list of dictionaries of the top k most similar embeddings and their similarity scores
        result = [{'id': int(id), 's_e_score': float(s_e_score), 'e_score': float(e_score)} for id, s_e_score, e_score in zip(top_k_ids, top_k_scores_scaled,top_k_scores)]

        # Turn id into int
        for r in result:
            r["id"] = int(r["id"])

        return result
    
    def query(self, w_ids, query, n):
        # Get the embedding of the query
        #query_embed = self.embed(query)
        
        query_embed = embedding = np.random.rand(1536)

        # Weigt the embeddings
        weighted_average_embed = self.embed_weighting(query_embed, w_ids)
        
        # Get the top n ids
        top_n_results = self.embeddings_knn(weighted_average_embed, n)
        
        return top_n_results