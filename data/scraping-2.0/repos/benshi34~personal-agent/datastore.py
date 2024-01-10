import openai
import pinecone
from prompts import INFO_AGENT_PROMPT
import os
import time
import json
import itertools
from uuid import uuid4
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

from dotenv import load_dotenv

load_dotenv(override=True)

openai.api_key = os.environ.get('OPENAI_API_KEY')

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENVIRONMENT')

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

EMB_MODEL = "text-embedding-ada-002"

INDEX_PREFIX = 'personal-agent'

EMBED_DIM = 1536 # text embedding vector size


class PineconeDatastore:
    def __init__(self, user):

        self.user = user

        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENV
        )

        self.INDEX_NAME = f"{INDEX_PREFIX}-{self.user.lower()}"

        if self.INDEX_NAME not in pinecone.list_indexes():
            print(f"Creating {self.INDEX_NAME} index from scratch")
            pinecone.create_index(
                name=self.INDEX_NAME, 
                dimension=EMBED_DIM,
                metric='cosine',
                pods=1
            )

            # wait for index to be fully initialized
            time.sleep(1)
        
        self.index = pinecone.Index(self.INDEX_NAME)

        print("Pinecone Datastore initialized!")

    # gets the top-k matches and organizes them into one string
    def read(self, text, k = None):
        
        # If k is None, then dynamically determin the k nearest neighbors to
        # include in contest
        matches = None

        if k is None:
            matches = self._index_query(text, k=100, include_values=False)['matches']
            
            if len(matches) == 0:
                return "There are no memories."
            
            # should be sorted in descending order
            scores = [match['score'] for match in matches] 

            indexes_keep = _determine_indices_kde(scores)

            matches = [matches[i] for i in indexes_keep]
        
        else:
            matches = self._index_query(text, k=k, include_values=False)['matches']
            if len(matches) == 0:
                return "There are no memories."
        
        # create context given a set of matches
        context = ""

        for match in matches:
            context += match['metadata']['text'] + '\n'

        return context
    
    # similar to read, but provides the matches list
    # can access scores through match['score']
    # can access text through match['metadata']['text']
    def _index_query(self, text, k, include_values):
        query_embedding = _get_text_embedding(text)

        res = self.index.query(
                                [query_embedding], 
                                top_k=k, 
                                include_metadata=True,
                                include_values=include_values
                            )

        return res
    


    def write(self, text, datetime_obj = None):

        # preprend text with timestamp
        if datetime_obj == None:
            timestamp = datetime.now().strftime("%B %d, %Y, %H:%M:%S")
        else:
            timestamp = datetime_obj.strftime("%B %d, %Y, %H:%M:%S")

        text = f"{timestamp}: {text}"

        embedding = _get_text_embedding(text) # list rep of embedding

        metadata = {
            'user': self.user,
            'text': text
        }

        self.index.upsert(
            vectors=[
                (
                    str(uuid4()), # Vector ID
                    embedding, # Dense vector
                    metadata,
                )
            ]
        )

        return "Write complete!"
    
    # https://docs.pinecone.io/docs/insert-data
    def _chunks(self, iterable, batch_size=100):
        """A helper function to break an iterable into chunks of size batch_size."""
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))
    
    def batch_write(self, texts, datetime_objs = None):
        vectors = []

        for i in range(len(texts)):
            
            text = texts[i]

            # preprend text with timestamp
            if datetime_objs == None:
                timestamp = datetime.now().strftime("%B %d, %Y, %H:%M:%S")
            else:
                timestamp = datetime_objs[i].strftime("%B %d, %Y, %H:%M:%S")

            text = f"{timestamp}: {text}"

            embedding = _get_text_embedding(text) # list rep of embedding

            vectors.append(
                (
                    str(uuid4()), # Vector ID
                    embedding, # Dense vector
                    {
                        'user': self.user,
                        'text': text
                    },
                )
            )
        
        for ids_vectors_chunk in self._chunks(vectors):
            self.index.upsert(vectors=ids_vectors_chunk)
    

    # https://stackoverflow.com/questions/75894927/pinecone-can-i-get-all-dataall-vector-from-a-pinecone-index-to-move-data-i
    def get_ids_from_query(self, input_vector):
        results = self.index.query(vector=input_vector,
                            top_k=10000, include_values=False)
        ids = set()
        for result in results['matches']:
            ids.add(result['id'])
        return ids

    # https://stackoverflow.com/questions/75894927/pinecone-can-i-get-all-dataall-vector-from-a-pinecone-index-to-move-data-i
    def get_all_ids_from_index(self, namespace=""):
        #print(index.describe_index_stats())
        num_vectors = self.index.describe_index_stats()["total_vector_count"]#[namespace]['vector_count']
        all_ids = set()
        while len(all_ids) < num_vectors:
            
            input_vector = np.random.rand(EMBED_DIM).tolist()
            
            ids = self.get_ids_from_query(input_vector)
            all_ids.update(ids)
            

        return list(all_ids)
    
    def get_all_memories(self):
        all_ids = self.get_all_ids_from_index()

        if len(all_ids) == 0:
            return []

        resp = self.index.fetch(ids=all_ids)

        mems = []

        for record in list(resp['vectors'].values()):
            mems.append(record['metadata']['text'])
        
        return mems


    # Only call if you want to delete the entire index!
    # Can't be reversed
    def delete_index(self):
        if self.INDEX_NAME not in pinecone.list_indexes():
            print("Index already deleted!")
        pinecone.delete_index(self.INDEX_NAME)
        print("Successfully deleted index!")
    
    # Does not delete the index, just erases all the memories in the index
    def delete_vectors(self):
        self.index.delete(delete_all=True)

# Class for doing primitive KNN and datastore
# Store data locally and load from existing files
class PrimitiveDatastore:
    def __init__(self, user, file=None) -> None:
        # list of dicts with fields: text, user, embedding
        self.dataset = []
        self.user = user

        if file is not None:
            f = open(file)
            self.dataset = json.load(f)


    def read(self, text, k = None):

        if(len(self.dataset) == 0):
            return ""

        query_embedding = _get_text_embedding(text)

        scores = np.array([cosine_similarity(query_embedding, sample['embedding']) for sample in self.dataset])

        asc_order = np.argsort(scores)

        desc_order = np.flip(asc_order)

        matches = None

        if k is None:
            # keep top 100 to consider relevance
            desc_order = desc_order[0:min(len(desc_order),100)]

            matches = [self.dataset[i] for i in desc_order]
            scores = scores[desc_order]

            indexes_keep = _determine_indices_kde(scores)

            matches = [matches[i] for i in indexes_keep]
        else:
            desc_order = desc_order[0:min(len(desc_order),k)]
            matches = [self.dataset[i] for i in desc_order]

        # create context given a set of matches
        context = ""

        for match in matches:
            context += match['text'] + '\n'

        return context


    def write(self, text):
        # preprend text with timestamp
        timestamp = datetime.now().strftime("%B %d, %Y, %H:%M:%S")
        text = f"{timestamp}: {text}"

        embedding = _get_text_embedding(text) # list rep of embedding

        record = {
            'text': text,
            'user': self.user,
            'embedding': embedding,
        }

        self.dataset.append(record)
    
    def get_all_memories(self):

        mems = [record['text'] for record in self.dataset]

        return mems


    def save_dataset(self, path):
        out_file = open(path, 'w')
        json.dump(self.dataset, out_file)


# Static helper methods below
# -----------------------------------------

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

def _get_text_embedding(text):
        # Get embedding for text

        res = openai.Embedding.create(
            input=[text],
            model=EMB_MODEL
        )

        return res['data'][0]['embedding'] # list rep of embedding

# use manual cosine similarity cutoff to determine items
# include in context
def _determine_indices_manual(scores, cutoff):

    indices = np.array(range(len(scores)))
    scores = np.array(scores)

    ret = indices[scores > cutoff].tolist()


    # if no scores meet cutoff just return index for highest score
    if len(ret) == 0:
        return [int(np.argmax(scores))]
    
    return ret

# Use kmeans with self-supplied value for k
# assume k is between 2 and 5 for most data
def _determine_indices_kmeans(scores, k=3):
    indices = np.array(range(len(scores)))

    X = np.array([scores]).T
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(X)
    max_label = np.argmax(kmeans.cluster_centers_)

    mask = kmeans.labels_.flatten() == max_label

    return indices[mask].tolist()

# https://stackoverflow.com/questions/35094454/how-would-one-use-kernel-density-estimation-as-a-1d-clustering-method-in-scikit
# It's not obvious what the optimal bandwidth is for KDE
def _determine_indices_kde(scores):

    X = np.array([scores]).T

    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)

    s = np.linspace(0,1,50)

    # kde produces log likelihood so we apply exp transform
    pdf = np.exp(kde.score_samples(s.reshape(-1,1)))

    mi = argrelextrema(pdf, np.less)[0]

    # edge case, no relative extrema found
    if len(mi) == 0:
        # restrict look to top 20% in cumulative prob
        # rescale so that cumulative prob sums to 1
        pdf = pdf/np.sum(pdf)

        threshold_value = 0.8

        ind = np.where(np.cumsum(pdf) > threshold_value)[0][0]

        cutoff = s[ind]
    # use largest rel min as cutoff
    else: 
        cutoff = np.max(s[mi])

    return _determine_indices_manual(scores, cutoff)