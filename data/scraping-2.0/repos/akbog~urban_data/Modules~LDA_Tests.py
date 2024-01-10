import pyarrow.parquet as pq
import pyarrow as pa
import dask.dataframe as dd
from dask import delayed
from pyarrow.parquet import read_schema, ParquetDataset
from Urban_Research.urban_data.Modules.LdaPipeline import GensimTopicModels
from gensim.models.coherencemodel import CoherenceModel
import os


PARAMETER_LIST = [
    {
        "n_topics" : 20,
        "update_every" : 5000,
        "passes" : 2,
        "alpha" : "auto",
        "scorer" : "mass_u",
        "include_bigram" : True,
    },
    {
        "n_topics" : 40,
        "update_every" : 5000,
        "passes" : 2,
        "alpha" : "auto",
        "scorer" : "mass_u",
        "include_bigram" : True,
    },
    {
        "n_topics" : 80,
        "update_every" : 5000,
        "passes" : 2,
        "alpha" : "auto",
        "scorer" : "mass_u",
        "include_bigram" : True,
    },
    {
        "n_topics" : 120,
        "update_every" : 5000,
        "passes" : 2,
        "alpha" : "auto",
        "scorer" : "mass_u",
        "include_bigram" : True,
    },
    {
        "n_topics" : 160,
        "update_every" : 5000,
        "passes" : 2,
        "alpha" : "auto",
        "scorer" : "mass_u",
        "include_bigram" : True,
    },
    {
        "n_topics" : 200,
        "update_every" : 5000,
        "passes" : 2,
        "alpha" : "auto",
        "scorer" : "mass_u",
        "include_bigram" : True,
    }
]

if __name__ == "__main__":
    DATA_DIR = "Tweet_Directory/DFST/"
    # os.environ.update({'MALLET_HOME':r'Mallet/'})
    # mallet_path = 'Mallet/bin/mallet'
    filters = [[("tw_year", "=", 2020), ("tw_month", "=", 4), ("tw_day", "=", 1)], [("tw_year", "=", 2020), ("tw_month", "=", 4), ("tw_day", "=", 2)]]
    parquet_pandas_df = pq.ParquetDataset(DATA_DIR, filters = filters).read_pandas()
    docs = [list(token) for token in parquet_pandas_df.to_pandas().tokens]
    stopwords = ['coronavirus', 'virus', 'pandemic', 'people', '$', "+", "|"]
    for test in PARAMETER_LIST:
        gensim_lda = GensimTopicModels(n_topics = test["n_topics"], update_every = test["update_every"],
                                        passes = test["passes"], alpha = test["alpha"],
                                        scorer = test["scorer"], include_bigram = test["include_bigram"],
                                        bigram_path = "TestingTopics/{}_bigram_model.pkl".format(test["n_topics"]),
                                        stopwords = stopwords)
        lda = gensim_lda.model.named_steps['model'].gensim_model
        lda.save("TestingTopics/{}_Bigram_Model.model".format(test["n_topics"]))
        lexicon = gensim_lda.model.named_steps['vect'].id2word
        corpus = [
            gensim_lda.model.named_steps['vect'].id2word.doc2bow(doc)
            for doc in gensim_lda.model.named_steps['norm'].transform(docs)
            ]
        #Coherence Scores 'u_mass', 'c_v', 'c_uci', 'c_npmi'
        coherence_scores = {}
        for coh_score in ['u_mass', 'c_v', 'c_uci', 'c_npmi']:
            coherence = CoherenceModel(model = lda, corpus = corpus,
                                        dictionary = lexicon, texts = docs,
                                        coherence = coh_score)
            coherence_scores["{} {}".format(test["n_topics"], coh_score)] = coherence.get_coherence()
        
