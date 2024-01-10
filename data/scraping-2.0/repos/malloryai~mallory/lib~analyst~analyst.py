import json
from attr import dataclass
from typing import List
import sys

#from google.cloud import pubsub_v1

import openai
import time
from datetime import datetime

# Langchain
import tiktoken
from langchain.llms import OpenAI
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    PromptTemplate,
    #ChatPromptTemplate,
    #SystemMessagePromptTemplate,
    #AIMessagePromptTemplate,
    #HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# config 
from config.config import Config

# Local application/library specific imports
from lib.news_analysis_result import *
from lib.analyst.prompts import *

# different analyst types 
from lib.analyst.data_breach import DataBreachNewsAnalyst, DataBreachNewsAnalysisResult
from lib.analyst.malware import MalwareNewsAnalyst, MalwareNewsAnalysisResult
from lib.analyst.threat import ThreatNewsAnalyst, ThreatNewsAnalysisResult
from lib.analyst.triage import TriageNewsAnalyst, TriageNewsAnalysisResult
from lib.analyst.vulnerability import VulnerabilityNewsAnalyst, VulnerabilityNewsAnalysisResult
from lib.analyst.industry_funding_announcement import IndustryFundingAnnoucementNewsAnalyst, IndustryFundingAnnouncementNewsAnalysisResult
from lib.analyst.patch import PatchNewsAnalyst, PatchNewsAnalysisResult
from lib.analyst.general_education import GeneralEducationNewsAnalyst, GeneralEducationNewsAnalysisResult


class Analyst:

    def cleanse_output(self, output: str) -> str:
        try:
            # TODO... remove 
            # remove "The output for this content should be:"
            output = output.replace("The output for this content should be:", "")
            output = output.replace("The output should be:", "")
            output = output.replace("The output would be:", "")
            output = output.replace("The output instance should be:", "")
            output = output.replace("The output should be formatted as a JSON instance:", "")
            output = output.replace("The output JSON instance should be:", "")
        
            # remove any text preceding a { character from output        
            #output = output[output.index("{"):]
            
            # remove any text after the last } character from output
            #output = output[:output.rindex("}") + 1]
        
            # remove newlines
            output = output.replace("\n", "")
        
        except Exception as e:
            raise Exception(f"Error occurred while cleansing output: {e}")
        
        return output

    def openai_analyze_news_event(self, newsItem, prompt_generator, output_parser):
        
        # use an output parser, since we'll get JSON back
        parser = PydanticOutputParser(pydantic_object=output_parser)

        # set up our prompts 
        prompt_template = PromptTemplate(
            template="\n{format_instructions}\n{p}\n",
            input_variables=["p"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # grab our text by passing in a news item for analysis 
        prompt_text = prompt_generator(url=newsItem.url, content=newsItem.Content)

        # and now we have it with all the formatting we want 
        input = prompt_template.format(p=prompt_text)
   
        messages = [
            SystemMessage(
                content="You are a world renowned threat researcher. Your objective is to analyze content, answer questions and output a JSON file with the result."
            ),
            HumanMessage(
                content=input
            ),
        ]

        model_name = "gpt-3.5-turbo-16k"
        limit_tokens = 16000
        temperature = 0.0

        # text-davinci-003 model has a total token limit of 4097
        encoding = tiktoken.encoding_for_model(model_name)
        prompt_tokens = len(encoding.encode(input))

        #print("DEBUG: Submitting request to ChatOpenAI with {} tokens".format(prompt_tokens), file=sys.stderr)

        # model = OpenAI(model_name="text-ada-001", n=2, best_of=2)
        chat = ChatOpenAI(model_name=model_name, temperature=temperature, max_retries=1, max_tokens=limit_tokens-prompt_tokens)
        
        # TODO ... this doesnt seem to work? 
        #chat.openai_api_key = Config.MALLORY_OPENAI_API_KEY
        
        # do the thing
        while True:
            try:
                response = chat(messages)
                break  # Break out of the loop if no exception occurs
            except openai.error.RateLimitError:
                print("WARNING! Rate limit exceeded. Retrying after backoff...", file=sys.stderr)
                time.sleep(60)  # Wait for 1 minute
                continue  # Continue to the next iteration of the loop

        # grab just the content ... TODO .. response has additional metrics to look at 
        output = response.content
        
        # clean up common errors in the output ... TEMPORARILY DISABLED 
        # cleansed_output = self.cleanse_output(output)

        # might be nil 
        analysis_result = None

        # Test that it's valid JSON 
        try:
            json.loads(output)
            # Okay, parse it up 
            analysis_result = parser.parse(output)
            
        except json.JSONDecodeError:
            print("WARNING! Output is NOT a valid object: {}".format(output))

            for _ in range(3):
                try: 
                    # do a one-time fix using the llm 
                    # output fixing parser will try to fix this 
                    fix_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
                    analysis_result = fix_parser.parse(output)
                    break
                except Exception: # TODO.... should be parsing exception?
                    print("WARNING! Output is NOT a valid object, trying output fixing parser!")

        return analysis_result


    def triage_news_event(self, newsItem) -> TriageNewsAnalysisResult:
        nar = self.openai_analyze_news_event(newsItem, TriageNewsAnalyst.generate_prompt, TriageNewsAnalysisResult)
        nar.url = newsItem.url
        nar.calculate_hash_values()
        nar.analysis_type = "triage_analysis"
        return nar 
    
    def analyze_vulnerability_news_event(self, newsItem) -> VulnerabilityNewsAnalysisResult:
        nar = self.openai_analyze_news_event(newsItem, VulnerabilityNewsAnalyst.generate_prompt, VulnerabilityNewsAnalysisResult)
        nar.url = newsItem.url
        nar.calculate_hash_values()
        nar.analysis_type = "vulnerability_analysis"
        return nar

    def analyze_threat_news_event(self, newsItem)  -> ThreatNewsAnalysisResult:
        nar = self.openai_analyze_news_event(newsItem, ThreatNewsAnalyst.generate_prompt, ThreatNewsAnalysisResult)
        nar.url = newsItem.url
        nar.calculate_hash_values()
        nar.analysis_type = "threat_analysis"
        return nar

    def analyze_malware_news_event(self, newsItem)  -> MalwareNewsAnalysisResult:
        nar = self.openai_analyze_news_event(newsItem, MalwareNewsAnalyst.generate_prompt, MalwareNewsAnalysisResult)
        nar.url = newsItem.url
        nar.calculate_hash_values()
        nar.analysis_type = "malware_analysis"
        return nar

    def analyze_data_breach_news_event(self, newsItem)  -> DataBreachNewsAnalysisResult:
        nar = self.openai_analyze_news_event(newsItem, DataBreachNewsAnalyst.generate_prompt, DataBreachNewsAnalysisResult)
        nar.url = newsItem.url
        nar.calculate_hash_values()
        nar.analysis_type = "data_breach_analysis"
        return nar

    def analyze_industry_funding_announcement_news_event(self, newsItem)  -> IndustryFundingAnnouncementNewsAnalysisResult:
        nar = self.openai_analyze_news_event(newsItem, IndustryFundingAnnoucementNewsAnalyst.generate_prompt, IndustryFundingAnnouncementNewsAnalysisResult)
        nar.url = newsItem.url
        nar.calculate_hash_values()
        nar.analysis_type = "industry_funding_announcement_analysis"
        return nar
    
    def analyze_patch_release_news_event(self, newsItem)  -> PatchNewsAnalysisResult:
        nar = self.openai_analyze_news_event(newsItem, PatchNewsAnalyst.generate_prompt, PatchNewsAnalysisResult)
        nar.url = newsItem.url
        nar.calculate_hash_values()
        nar.analysis_type = "patch_release_analysis"
        return nar

    def analyze_general_education_news_event(self, newsItem)  -> GeneralEducationNewsAnalysisResult:
        nar = self.openai_analyze_news_event(newsItem, GeneralEducationNewsAnalyst.generate_prompt, GeneralEducationNewsAnalysisResult)
        nar.url = newsItem.url
        nar.calculate_hash_values()
        nar.analysis_type = "general_education_analysis"
        return nar
    