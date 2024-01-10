"""
Topic Modeling with gensim: Evaluation of model quality.

Provides model quality indicators at the model and topic level.

See: https://radimrehurek.com/gensim/models/coherencemodel.html
"""

# == Imports ==

import pandas as pd
from os.path import join
from gensim import corpora
from gensim import models
from gensim.models.coherencemodel import CoherenceModel
import helpers


# == Functions == 

def model_coherence(listcorpus, vectorcorpus, model, numtopics, resultsfolder): 
    """
    Calculate coherence scores for the entire model, 
    using several different measures. 
    """
    print("model_coherence")
    measures = ["c_v", "c_npmi", "u_mass", "c_uci"]
    coherences = []
    for measure in measures: 
        coherencemodel = CoherenceModel(texts=listcorpus, model=model, corpus=vectorcorpus, coherence=measure, processes=3)
        coherence = coherencemodel.get_coherence()
        coherences.append(coherence)
    coherences = dict(zip(measures, coherences))
    coherences = pd.DataFrame.from_dict(coherences, orient='index', columns=["score"])
    with open(join(resultsfolder, "coherences-model.csv"), "w", encoding="utf8") as outfile: 
        coherences.to_csv(outfile, sep="\t")


def topic_coherence(listcorpus, vectorcorpus, model, numtopics, resultsfolder): 
    """
    Calculate coherence scores for each topic, using one measure only.
    """
    print("topic_coherence")
    coherencemodel = CoherenceModel(texts=listcorpus, model=model, corpus=vectorcorpus, coherence="c_v", processes=3)    
    coherences = list(zip(range(0,numtopics), coherencemodel.get_coherence_per_topic()))
    coherences = pd.DataFrame(coherences, columns=["topic", "score"]).sort_values(by="score", ascending=False)
    with open(join(resultsfolder, "coherences-topics.csv"), "w", encoding="utf8") as outfile: 
        coherences.to_csv(outfile, sep="\t")


# == Coordinating function == 

def main(workdir, identifier, numtopics): 
    print("\n== evaluation ==")
    listcorpus = helpers.load_pickle(workdir, identifier, "allprepared.pickle")
    vectorcorpus = helpers.load_pickle(workdir, identifier, "vectorcorpus.pickle")
    model = helpers.load_model(workdir, identifier)
    resultsfolder = join(workdir, "results", identifier)
    model_coherence(listcorpus, vectorcorpus, model, numtopics, resultsfolder)
    topic_coherence(listcorpus, vectorcorpus, model, numtopics, resultsfolder)
    print("\nDone.\n")
    

