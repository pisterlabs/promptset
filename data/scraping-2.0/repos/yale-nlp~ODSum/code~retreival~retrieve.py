import json
import os
import numpy as np
import math
import torch
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split

from sklearn.metrics.pairwise import cosine_similarity


USE_CONTRIEVER = True
USE_BM25 = False
USE_LLM_EMBEDDING = False



def get_embedding(text, model="text-embedding-ada-002", max_tokens=4000):
    """
    Fetches embeddings for the given text using OpenAI.
    If the text exceeds the maximum token limit, it breaks the text into chunks.
    """
    text = text.replace("\n", " ")

    # Tokenize the text to count tokens (this is a simplistic example; a more accurate method could be used)
    tokens = text.split()

    if len(tokens) <= max_tokens:
        # If the text fits within the token limit, get the embedding as usual
        return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
    else:
        # If the text exceeds the token limit, break it into chunks
        chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]

        # Convert chunks back to string
        chunks = [" ".join(chunk) for chunk in chunks]

        # Get embeddings for each chunk
        chunk_embeddings = np.array([openai.Embedding.create(input=[chunk], model=model)['data'][0]['embedding'] for chunk in chunks])

        # Calculate weights based on the number of tokens in each chunk
        weights = [len(chunk.split()) for chunk in chunks]

        # Calculate the weighted average of the embeddings
        weighted_avg_embedding = np.average(chunk_embeddings, axis=0, weights=weights)

        return weighted_avg_embedding


class InformationRetrieval:
    def __init__(self, query_file, meeting_folder, embedding_file='precomputed_embeddings.json', contriever_embedding_file='contriever_precomputed_embeddings.json'):
        with open(query_file, 'r') as file:
            self.queries_dict = json.load(file)



        self.results = {'MIN': [], 'MEAN': [], 'MAX': []}
        self.performance_metrics = {'MIN': [], 'MEAN': [], 'MAX': []}

        self.meeting_ids = []
        self.meetings = []



        for filename in sorted(os.listdir(meeting_folder)):
            meeting_id = filename.split('.')[0]  # Assuming filename is like "TS3009d.txt"
            self.meeting_ids.append(meeting_id)

            with open(os.path.join(meeting_folder, filename), 'r') as file:
                self.meetings.append(file.read())

        # Read meetings and create a mapping from meeting IDs to texts
        self.meeting_texts = {}
        for filename in os.listdir(meeting_folder):
            meeting_id = filename.split('.')[0]
            with open(os.path.join(meeting_folder, filename), 'r') as file:
                self.meeting_texts[meeting_id] = file.read()

        if USE_CONTRIEVER:
            from src.contriever import Contriever
            from transformers import AutoTokenizer

            # Initialize Contriever and tokenizer
            self.contriever = Contriever.from_pretrained("facebook/contriever")
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
            # Check if a GPU is available and if not, fall back to CPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Move the model to the device
            self.contriever.to(self.device)

             # Try to load precomputed embeddings
            try:
                with open(contriever_embedding_file, 'r') as f:
                    self.embeddings = json.load(f)
                print("Loaded precomputed embeddings from file.")
            except FileNotFoundError:
                print("No precomputed embeddings found. Computing now...")
                self.embeddings = self.precompute_contriever_embeddings()
                with open(contriever_embedding_file, 'w') as f:
                    json.dump(self.embeddings, f)
                print(f"Saved precomputed embeddings to {contriever_embedding_file}.")

        if USE_BM25:
            from rank_bm25 import BM25Okapi

        if USE_LLM_EMBEDDING:
        # Try to load precomputed embeddings from file
            import openai
            openai.api_key = '' # Put your OpenAI API key here
            try:
                with open(embedding_file, 'r') as f:
                    self.embeddings = json.load(f)
                print("Loaded precomputed embeddings from file.")
            except FileNotFoundError:
                print("No precomputed embeddings found. Computing now...")
                self.embeddings = self.precompute_embeddings()  # Dictionary to store pre-computed embeddings
                # Save the computed embeddings to a file
                with open(embedding_file, 'w') as f:
                    json.dump(self.embeddings, f)
                print(f"Saved precomputed embeddings to {embedding_file}.")


    def contriever_embedding(self, query, n):
         # Tokenize and get embeddings for the query
        query_inputs = self.tokenizer([query], padding=True, truncation=True, return_tensors="pt")
        query_inputs = {key: tensor.to(self.device) for key, tensor in query_inputs.items()}

        with torch.no_grad():
            query_output = self.contriever(**query_inputs).to(self.device)  # Ensure it's on the same device

        query_embedding = query_output.cpu().numpy().reshape(1, -1)  # Move to CPU for numpy operations

        # Assume doc_embeddings are initially on CPU
        doc_embeddings = np.array([self.embeddings[meeting_id] for meeting_id in self.meeting_ids])

        # Calculate similarity scores using cosine similarity
        doc_scores = cosine_similarity(query_embedding, doc_embeddings)[0]

        # Sort by similarity score and select top n indices
        top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:n]

        # Get the corresponding top n meeting IDs
        top_n_meeting_ids = [self.meeting_ids[i] for i in top_n_indices]

        return top_n_meeting_ids

    def precompute_embeddings(self):
        embeddings = {}
        for meeting_id, meeting_text in tqdm(zip(self.meeting_ids, self.meetings), desc='Precomputing embeddings', total=len(self.meetings)):
            # Assuming get_embedding function is available and returns a NumPy array
            embeddings[meeting_id] = get_embedding(meeting_text)
        return embeddings

    def precompute_contriever_embeddings(self):
        embeddings = {}
        for meeting_id, meeting_text in zip(self.meeting_ids, self.meetings):
            # Tokenize and get embeddings for each meeting
            inputs = self.tokenizer([meeting_text], padding=True, truncation=True, return_tensors="pt")
            inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
            with torch.no_grad():
                outputs = self.contriever(**inputs)
            # Save the computed embeddings
            embeddings[meeting_id] = outputs.cpu().numpy()[0].tolist()
        return embeddings


    def llm_embedding(self, query, n):
         # Compute the query embedding
        query_embedding = np.array(get_embedding(query)).reshape(1, -1)

        # Create a 2D array for document embeddings
        doc_embeddings = np.array([self.embeddings[meeting_id] for meeting_id in self.meeting_ids])

        # Calculate similarity scores
        doc_scores = cosine_similarity(query_embedding, doc_embeddings)[0]

        # Sort by similarity score and select top n indices
        top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:n]

        # Get the corresponding top n meeting IDs
        top_n_meeting_ids = [self.meeting_ids[i] for i in top_n_indices]

        return top_n_meeting_ids

    def bm25(self, query, n):
        tokenized_meetings = [doc.split(" ") for doc in self.meetings]
        tokenized_query = query.split(" ")
        bm25 = BM25Okapi(tokenized_meetings)
        doc_scores = bm25.get_scores(tokenized_query)

        # print("Debug: len(doc_scores):", len(doc_scores))
        # print("Debug: len(self.meeting_ids):", len(self.meeting_ids))
        # print("Debug: len(self.meetings):", len(self.meetings))

        top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:n]



        top_n_meeting_ids = [self.meeting_ids[i] for i in top_n_indices]
        return top_n_meeting_ids



    # Function to calculate performance metrics
    def evaluate_performance(self, retrieved_meetings, ground_truth_meetings):
        retrieved_set = set(retrieved_meetings)
        ground_truth_set = set(ground_truth_meetings)

        tp = len(retrieved_set.intersection(ground_truth_set))
        fp = len(retrieved_set) - tp
        fn = len(ground_truth_set) - tp

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        # NDCG calculation
        dcg = 0
        ideal_dcg = 0
        for i, doc in enumerate(retrieved_meetings):
            rel = 1 if doc in ground_truth_set else 0
            dcg += rel / math.log2(i + 2)  # i starts from 0, log starts from 2

        for i in range(min(len(ground_truth_meetings), len(retrieved_meetings))):
            ideal_dcg += 1 / math.log2(i + 2)

        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0

        # MAP calculation (Average Precision for this query)
        num_relevant = 0
        sum_precision = 0
        for i, doc in enumerate(retrieved_meetings):
            if doc in ground_truth_set:
                num_relevant += 1
                sum_precision += num_relevant / (i + 1)

        ap = sum_precision / len(ground_truth_set) if ground_truth_set else 0

        return {'Precision': precision, 'Recall': recall, 'F1': f1, 'NDCG': ndcg, 'AP': ap}

    def save_splits(self, method_results, dir_path):
        # Create train, test and dev splits (6:3:1 ratio)
        train, temp = train_test_split(method_results, test_size=0.4, random_state=42)
        test, dev = train_test_split(temp, test_size=0.25, random_state=42)

        # # Create directories if they don't exist
        # for split in ['train', 'test', 'dev']:
        #     split_dir_path = os.path.join(dir_path, split)
        #     if not os.path.exists(split_dir_path):
        #         os.makedirs(split_dir_path)

        # Save the splits to their respective JSON files
        with open(f"{dir_path}/train.json", 'w') as f:
            json.dump(train, f, indent=4)
        with open(f"{dir_path}/test.json", 'w') as f:
            json.dump(test, f, indent=4)
        with open(f"{dir_path}/dev.json", 'w') as f:
            json.dump(dev, f, indent=4)

    def run_evaluation(self, methods, n_values=[1, 3, 6], labels=['MIN', 'MEAN', 'MAX']):
        # Initialize dictionary to store sum and count for each metric and label
        avg_performance_metrics = defaultdict(lambda: defaultdict(lambda: {'sum': 0.0, 'count': 0}))

        for n, label in zip(n_values, labels):
            for method in methods:
                # Initialize list to hold results for this method and top_k
                method_results = []

                for query_id, query_data in tqdm(self.queries_dict.items()):
                    query = query_data['query']
                    answer = query_data['answer']
                    ground_truth_meetings = query_data['meetings_list']

                    retrieved_meetings = getattr(self, method)(query, n)

                    # Evaluate performance
                    metrics = self.evaluate_performance(retrieved_meetings, ground_truth_meetings)

                    retrieved_texts = [self.meeting_texts[meeting_id] for meeting_id in retrieved_meetings]

                    # Prepare the result dictionary
                    result_dict = {
                        # 'query_id': query_id,
                        'Query': query,
                        'Summary': answer,
                        'Article': '<doc-sep>'.join(retrieved_texts),  # Assuming retrieved_meetings is a list of meeting texts
                        # 'Metrics': metrics
                    }

                    # Store in method_results
                    method_results.append(result_dict)

                    # Accumulate for average metrics
                    for metric_name, metric_value in metrics.items():
                        avg_performance_metrics[label][metric_name]['sum'] += metric_value
                        avg_performance_metrics[label][metric_name]['count'] += 1

                # Create directories if they don't exist
                dir_path = f"{method}/{label}"
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

                # # Save results to JSON file
                # with open(f"{dir_path}/results.json", 'w') as f:
                #     json.dump(method_results, f, indent=4)
                self.save_splits(method_results, dir_path)

        # Calculate and print average metrics
        for label, metrics in avg_performance_metrics.items():
            print(f"Average Performance Metrics for {label}:")
            for metric_name, values in metrics.items():
                avg_value = values['sum'] / values['count']
                print(f"{metric_name}: {avg_value:.4f}")
            print()



if __name__ == '__main__':
    ir = InformationRetrieval('updated_merged_queries.json', 'meetings/txt')
    ir.run_evaluation(['contriever_embedding'])