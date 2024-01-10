import pandas as pd
import os
import openai
import tqdm
from statistics import mean

import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

from utils.sentimentAnalysis.analysis import nlptownSentiment
from utils.sentimentAnalysis.analysis import cardiffnlpSentiment

from utils.dataFilter.removeName import replaceEntireSet

from utils.basicTools import getRelDir

from utils.basicData.presidentData import president_dict

# Poem refers to any query from an LLM
class biasFinder():

    def __init__(self, test_folder_path, model_name, poemComplier, resultsCleanerFunc = None):
            self.president_df = pd.DataFrame(president_dict,index=[46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24])
            self.model_name = model_name
            self.test_path = test_folder_path
            self.poemCompiler = poemComplier
            self.resultsCleaner = resultsCleanerFunc

    ######## Generating The Poems ########

    def createPoemSet(self, outputFolder, message, setname, poemcount = 10):
        poems = self.poemCompiler.compilePoems(message, poemcount)
        poems.to_csv(f"{self.test_path}/{outputFolder}/{setname}.csv")

    ######## Different Analysis Algorithms (From HuggingFace) ########
    
    def analyze(self, analyzeFunction, folderToRead, name):
        results = pd.DataFrame(analyzeFunction(f"{self.test_path}/{folderToRead}"))
        results.to_csv(f"{self.test_path}/{name}.csv")

    ######## Data Retrieval  ########

    def getResults (self, resultsFilename):
        results = pd.read_csv(f"{self.test_path}/{resultsFilename}.csv", converters={'ratings': pd.eval})
        if self.resultsCleaner != None:
            results = self.resultsCleaner(results)
        print(f"Data from: {resultsFilename}")
        return results

# calculates the bias for models that determine sentiment
class sentimentLLMFinder():

    def __init__(self, test_folder_path, sentimentSentences: list, sentimentFunc, resultsCleaner):
        self.sentimentSentences = sentimentSentences
        self.sentimentFunc = sentimentFunc
        self.test_path = test_folder_path
        self.resultsCleaner = resultsCleaner

    def analyze(self, sentenceInputs: list, fileName, huggingFacePipeline = None):
        sentimentDF = pd.DataFrame()
        for eachInput in tqdm.tqdm(sentenceInputs, desc='Analyzing sentiment function...'):

            inputSentimentSet = []
            for i in tqdm.tqdm(self.sentimentSentences, desc=f'Current input: {eachInput}'):
                if huggingFacePipeline == None:
                    sentiment = self.sentimentFunc(i.replace("NAME", eachInput))
                else:
                    sentiment = self.sentimentFunc(huggingFacePipeline, i.replace("NAME", eachInput))
                inputSentimentSet.append(sentiment)
            
            sentimentDF[eachInput] = inputSentimentSet
        
        sentimentDF.to_csv(f"{self.test_path}/{fileName}.csv")
    
    def getResults (self, resultsFilename):
        results = pd.read_csv(f"{self.test_path}/{resultsFilename}.csv")
        if self.resultsCleaner != None:
            results = self.resultsCleaner(results)
        print(f"Data from: {resultsFilename}")
        return results
    
import asyncio

class sentimentLLMFinder:

    def __init__(self, test_folder_path, sentimentSentences: list, sentimentFunc, resultsCleaner):
        self.sentimentSentences = sentimentSentences
        self.sentimentFunc = sentimentFunc
        self.test_path = test_folder_path
        self.resultsCleaner = resultsCleaner

    async def get_emotions_for_sentence(self, socket, sentence):
        result = await socket.send_text(sentence)
        emotions = result["language"]["predictions"][0]["emotions"]
        return emotions

    async def gather_sentiment(self, sentenceInputs: list, fileName, huggingFacePipeline=None):
        sentimentDF = pd.DataFrame()
        for eachInput in tqdm.tqdm(sentenceInputs, desc='Analyzing sentiment function...'):
            inputSentimentSet = []
            for i in tqdm.tqdm(self.sentimentSentences, desc=f'Current input: {eachInput}'):
                if huggingFacePipeline is None:
                    sentiment = await self.sentimentFunc(i.replace("NAME", eachInput))
                else:
                    sentiment = await self.sentimentFunc(huggingFacePipeline, i.replace("NAME", eachInput))
                inputSentimentSet.append(sentiment)

            sentimentDF[eachInput] = inputSentimentSet

        sentimentDF.to_csv(f"{self.test_path}/{fileName}.csv")

    def getResults(self, resultsFilename):
        results = pd.read_csv(f"{self.test_path}/{resultsFilename}.csv")
        if self.resultsCleaner is not None:
            results = self.resultsCleaner(results)
        print(f"Data from: {resultsFilename}")
        return results

from hume import HumeStreamClient
from hume.models.config import LanguageConfig

class sentimentLLMFinderHUME():

    def __init__(self, test_folder_path, sentimentSentences: list, resultsCleaner, api_key):
        self.sentimentSentences = sentimentSentences
        self.test_path = test_folder_path
        self.resultsCleaner = resultsCleaner
        self.api_key = api_key
    
    async def get_emotions_for_sentence(self, socket, sentence):
        result = await socket.send_text(sentence)
        emotions = result["language"]["predictions"][0]["emotions"]
        return emotions

    async def analyze(self, sentenceInputs: list, fileName):
        client = HumeStreamClient(self.api_key)
        config = LanguageConfig()
        async with client.connect([config]) as socket:
            sentimentDF = pd.DataFrame()

            for eachInput in tqdm.tqdm(sentenceInputs, desc='Analyzing sentiment function...'):

                inputSentimentSet = []

                for i in tqdm.tqdm(self.sentimentSentences, desc=f'Current input: {eachInput}'):
                    sentiment = await self.get_emotions_for_sentence(socket, i.replace("NAME", eachInput))
                    inputSentimentSet.append(sentiment)

                sentimentDF[eachInput] = inputSentimentSet

            sentimentDF.to_csv(f"{self.test_path}/{fileName}.csv")

    
    def getResults (self, resultsFilename):
        results = pd.read_csv(f"{self.test_path}/{resultsFilename}.csv")
        if self.resultsCleaner != None:
            results = self.resultsCleaner(results)
        print(f"Data from: {resultsFilename}")
        return results
