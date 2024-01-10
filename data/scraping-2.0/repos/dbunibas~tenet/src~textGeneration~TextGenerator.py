import openai
import json
import io
import os
import time
from src import Constants
from src.textGeneration import ColdStartExamples

class TextGenerator:

    def __init__(self):
        
        self.api_key_index = 0
        self.rateLimit = True
        self.useCache = True
        #self.cacheFile = "../../data/cache_GPT3.json"
        self.cacheFile = Constants.CACHE_DIR + "cache_GPT3.json"
        self.cache = {}
        self.enableGPT = False
        self.counterRequest = 0
        self.sleepTime = 0.1 #seconds
        if self.useCache and os.path.isfile(self.cacheFile):
            print("*** CACHE FILE: ", self.cacheFile)
            with open(self.cacheFile, 'r') as f:
                listPrompts = json.load(f)
            for item in listPrompts:
                self.cache[item['prompt']] = item['text']

    def generateText(self, prompt):
        text = None
        if self.useCache:
            text = self.cache.get(prompt)
        if text is None and self.enableGPT:
            openai.api_key = Constants.API_KEYS_GPT3[self.api_key_index]
            self.api_key_index = (self.api_key_index + 1) % len(Constants.API_KEYS_GPT3)
            response = openai.Completion.create(
                model="text-davinci-002",
                prompt=prompt,
                temperature=0,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            print(".", end="", flush=True)
            text = response["choices"][0]['text'].strip()
            self.counterRequest += 1
            if self.useCache:
                self.cache[prompt] = text
                toJson = []
                for key, value in self.cache.items():
                    toJson.append({"prompt": key, "text": value})
                jsonString = json.dumps(toJson, indent=4)
                with io.open(self.cacheFile, 'w') as f:
                    f.write(jsonString)
            if self.rateLimit:
                time.sleep(self.sleepTime) ## wait for seconds
        return text

    def linearizeEvidence(self, tableName, evidence, operator=None, attributeExtra=None, valueExtra=None):
        headers = evidence.headers
        prompt = "<table>" + tableName + "</table>\n"
        if attributeExtra is not None and valueExtra is not None:
            sValue = str(valueExtra)
            if (operator is not None) and (operator is not Constants.OPERATOR_SAME):
                sValue = " " + operator + " " + str(valueExtra)
            prompt += "<extra>" + sValue + " <header>" + attributeExtra + "</header>" + "</extra>"
        for header in headers:
            prompt += "<header>" + header.name + "</header>"
        prompt += "\n"
        for row in evidence.orderedRows:
            prompt += "<row>"
            for header in headers:
                cell = row[header.name]
                value = ""
                if cell is not None:
                    value = cell.value
                prompt += "<cell>" + str(value) + "</cell>"
            prompt += "</row>\n"
        prompt += "example:\n"
        return prompt

    def generateColdStartPrompt(self, tableName, evidence, operation, operator=None, attributeExtra=None, valueExtra=None):
        linearizedInput = self.linearizeEvidence(tableName, evidence, operator, attributeExtra, valueExtra)
        if operation == Constants.OPERATION_COMPARISON:
            if operator == Constants.OPERATOR_GT: return ColdStartExamples.COLD_START_COMPARISON_GT + linearizedInput
            if operator == Constants.OPERATOR_LT: return ColdStartExamples.COLD_START_COMPARISON_LT + linearizedInput
            if operator == Constants.OPERATOR_SAME: return ColdStartExamples.COLD_START_COMPARISON_EQ + linearizedInput
        if operation == Constants.OPERATION_GROUPING:
            if operator == Constants.OPERATOR_GT: return ColdStartExamples.COLD_START_GROUPING_COUNT_COMPARISONS + linearizedInput
            if operator == Constants.OPERATOR_LT: return ColdStartExamples.COLD_START_GROUPING_COUNT_COMPARISONS + linearizedInput
            if operator == Constants.OPERATOR_SAME: return ColdStartExamples.COLD_START_GROUPING_COUNT + linearizedInput
        if operation == Constants.OPERATION_LOOKUP: return ColdStartExamples.COLD_START_LOOKUP + linearizedInput
        if operation == Constants.OPERATION_MAX: return ColdStartExamples.COLD_START_MAX + linearizedInput
        if operation == Constants.OPERATION_MIN: return ColdStartExamples.COLD_START_MIN + linearizedInput
        if operation == Constants.OPERATION_COUNT: return ColdStartExamples.COLD_START_COUNT + linearizedInput
        return None


