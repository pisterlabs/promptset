import pandas as pd
import os
import openai
import tqdm

import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

from utils.dataRequest.generator import poemCompiler

from utils.sentimentAnalysis.analysis import nlptownSentiment
from utils.sentimentAnalysis.analysis import cardiffnlpSentiment

from utils.sentimentAnalysis.analysisWideScale import nlptownSentimentWide
from utils.sentimentAnalysis.analysisWideScale import cardiffnlpSentimentWide

from utils.dataFilter.removeName import replaceEntireSet

from utils.models.modelTemplates import biasFinder

from utils.dataRequest.chatGPT_3_5_query import queryGPT_3_5

from utils.basicData.presidentData import president_dict
president_df = pd.DataFrame(president_dict,index=[46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24])


def resultsCleaner(results):
    results = results.set_index([[46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24]])
    #results = results.drop(['Unnamed: 0'], axis=1)
    results['party'] = president_df['party']
    years = president_dict['year'].copy()
    years.reverse()
    results['year'] = years
    results = results.drop(['Unnamed: 0'], axis=1)
    return results

#initialize bias finding machine
descriptionBias = biasFinder("chatGPT3_descriptions", "chatGPT-3: Descriptions", poemCompiler(queryGPT_3_5), resultsCleaner)

def createDescriptions():
    for president_name in tqdm.tqdm(president_df['name'],desc='Presidents Descriptions'):
        descriptionBias.createDescriptionSet("politicalDescriptions", f"Write a 10 sentence description about {president_name}.", president_name, 100)

def analyzeNLP():
    descriptionBias.analyze(nlptownSentiment,"politicalDescriptionsNameless","nlptown_nameless")

def analyzeCAR():
    descriptionBias.analyze(cardiffnlpSentiment,"politicalDescriptionsNameless","cardiffnlp_nameless")

def analyzeNLPwide():
    descriptionBias.analyze(nlptownSentimentWide,"politicalDescriptionsNameless","nlptown_nameless_wide")

def analyzeCARwide():
    descriptionBias.analyze(cardiffnlpSentimentWide,"politicalDescriptionsNameless","cardiffnlp_nameless_wide")

def getResults (resultsFilename):
    return descriptionBias.getResults(resultsFilename)

def cleanNames():
    replaceEntireSet("chatGPT3_descriptions/politicalDescriptions","chatGPT3_descriptions/politicalDescriptionsNameless")