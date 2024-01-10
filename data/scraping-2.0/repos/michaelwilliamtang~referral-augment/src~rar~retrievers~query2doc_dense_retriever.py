import numpy as np
from sqlitedict import SqliteDict
import openai

from .dense_retriever import DenseRetriever
from .utils import AggregationType, SimilarityType

class Query2DocDenseRetriever(DenseRetriever):
    def __init__(self, *args, use_sql=False, openai_organization=None, openai_api_key=None, **kwargs):
        '''
        use_sql: use SQLite cache to reduce GPT-3 usage instead of in-memory dict
        openai_organization: string, OpenAI organization
        openai_api_key: string, OpenAI API key
        '''
        assert openai_organization is not None
        assert openai_api_key is not None
        openai.organization = openai_organization
        openai.api_key = openai_api_key

        # create query2doc cache to reduce LLM usage
        cache_fname = 'query2doc_cache.sqlite'
        if use_sql:
            self.query2doc_cache = SqliteDict(cache_fname)
        else:
            self.query2doc_cache = {}

        super().__init__(*args, **kwargs)

    def get_query2doc(self, query, verbose=True):
        # check cache for response
        if query in self.query2doc_cache:
            if verbose:
                print('Using cached value for query')
            return self.query2doc_cache[query]

        # if not in cache, generate
        prefix = 'Write the abstract of the paper that is cited as [CITATION] in the below passage: '
        response = openai.Completion.create(
            model='text-davinci-003',
            prompt=prefix + query,
            temperature=0.7,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=1,
            stop=['\n\n\n']
            )
        if verbose:
            print('Used GPT-3 to generate query')

        # cache response
        self.query2doc_cache[query] = response['choices'][0]['text']
        return response['choices'][0]['text']

    def retrieve(self, query, num_docs=10, similarity=SimilarityType.DOT, query_weight=0.8,
                 verbose=True):
        '''
        query: string
        num_docs: int, number of top documents to retrieve
        similarity: type of vector similarity (dot product or cosine)
        query_weight: float in [0, 1], to give to original query vs. query2doc pseudo-doc
        '''
        num_docs = min(num_docs, len(self.docs))

        # apply query2doc augmentation
        assert query_weight >= 0 and query_weight <= 1
        query2doc = self.get_query2doc(query, verbose=verbose)
        encoded_query = query_weight * self.encoder.encode(query, is_query=True) + (1 - query_weight) * self.encoder.encode(query2doc, is_query=False)

        # compute similarity
        sims = self.embeds @ encoded_query
        if similarity == SimilarityType.COSINE:
            # normalize by norms
            norms = np.linalg.norm(self.embeds, axis=1) * np.linalg.norm(encoded_query)
            sims /= norms

        if self.aggregation == AggregationType.SHORTEST_PATH:
            # since we want num_docs unique documents, we retrieve more, then filter duplicates
            num_docs_before_filter = min(num_docs * self.num_referrals, len(self.docs))
            idxs = np.argpartition(sims, -num_docs_before_filter)[-num_docs_before_filter:]
            idxs = idxs[np.argsort(sims[idxs])[::-1]] # descending
            docs_before_filter = self.docs[idxs]
            return list(dict.fromkeys(docs_before_filter))[:num_docs]

        # get top num_docs -- note that argpartition is linear but the top k are unsorted
        # thus, we argpartition and then we sort post hoc to get efficient + sorted top num_docs
        idxs = np.argpartition(sims, -num_docs)[-num_docs:]
        idxs = idxs[np.argsort(sims[idxs])[::-1]] # descending
        return self.docs[idxs]

    # if using SQLite cache, call this to close at end of session
    def close_cache(self):
        self.query2doc_cache.commit()
        self.query2doc_cache.close()
