import ReferenceCode.PromptGenerators as PromptGenerators
import openai
from langchain.llms import OpenAI
import time 

openaiApiKey = ''

class ProcessHandler:
    def __init__(self,promptFString):
        self.pG = PromptGenerators.PromptGenerator(promptFString)
        self.llm = OpenAI(temperature=0.5, openai_api_key=openaiApiKey)
    
    def generateAnswer(self, questionPromptString):
        formatedQuestion = self.pG.generatePrompt(questionPromptString)
        return self.llm(formatedQuestion)
    
class MADProcessHandler:
    def __init__(self,nrounds,nAgents, prePromptString, postPromptString):
        self.pG = PromptGenerators.MADPromptGenerator(prePromptString,postPromptString,nAgents)
        self.nrounds = nrounds
        self.llm = OpenAI(temperature=0.5, openai_api_key=openaiApiKey)
        self.answersRecords = []
        self.nAgents = nAgents

    def generateAnswer(self, questionSting, prePromptString=None, postPromptString=None):
        self.answersRecords.clear()
        if prePromptString != None:
            questionSting += prePromptString
        for i in range(self.nrounds):
            self.answersRecords.append([])
            for j in range(self.nAgents):
                if i !=0:
                    prompt = self.pG.generatePrompt(self.answersRecords, j,i,preInputs=questionSting)
                else:
                    prompt = questionSting
                answer = self.llm(prompt)
                self.answersRecords[i].append(answer)
        return self.answersRecords[self.nrounds-1]
                



