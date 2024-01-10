import openai
import numpy as np
import os
from dotenv import load_dotenv
import os
from collections import deque
from typing import Dict, List, Optional, Any
from geopy.geocoders import Nominatim
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


from langchain import LLMChain, OpenAI, PromptTemplate

from langchain.chains import RetrievalQA, LLMChain ,LLMCheckerChain

from langchain import OpenAI, SerpAPIWrapper, LLMChain


from newspaper import Config, Article, Source
from utils.Webscraper import Webscraper

import jsonschema
import logging

webscraper = Webscraper()


from .chains import (
    articleClassifier,
    locationExtractor,
    eventDetails,
)

from .tools import get_coordinates
from .utils import num_tokens_from_string, extract_json_schema
from colorlog import ColoredFormatter
import logging
formatter = ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

class outputValidator:
    """
    Validates json output of any llmchain, given that it's created with the create_structured_output_chain function
    Gives Feedback to the LLMChain to improve the output
    """
    @staticmethod
    def _getOutputSchemaMapping(LLMChain:LLMChain) -> dict:
        """Returns the supposed output schema of the given LLMChain 
        
        Example llm_kwargs:
        {'functions': [{'name': 'output_formatter',
   'description': 'Output formatter. Should always be used to format your response to the user.',
   'parameters': {'name': 'binary_classifier_article_schema',
    'description': 'Binary Classifier Schema for Article, 0 for False and 1 for True',
    'type': 'object',
    'properties': {'isDisruptionEvent': {'type': 'boolean'},
     'Reason': {'type': 'string'}},
    'required': ['isDisruptionEvent', 'Reason']}}],
    'function_call': {'name': 'output_formatter'}}
    """
        output_schema = LLMChain.llm_kwargs
        # Create a dictionary mapping with Key as Key and Value as type
        validator_json_schema = extract_json_schema(output_schema)
        return validator_json_schema
        
    @classmethod
    def validate(cls, LLMChain:LLMChain, output: dict) -> Tuple[bool, str]:
        """Validates the output of the given llmchain
        Args:
            JSON_SCHEMA (dict): The JSON Schema to validate
            output (dict): The output of the LLMChain
        Returns:
            Tuple[bool, str]: A tuple containing a boolean and a string. 
            The boolean is True if the output is valid according to the schema, False if not. 
            The string is the reason for the boolean value. If the boolean is True, the string will be empty.
            The string might be used to give feedback to LLMChain to improve the output.
        """
        # Check if the output is a dict
        if not isinstance(output,dict):
            return (False,"The output is not JSON object (dict)")
        # Get the output schema mapping
        JSON_SCHEMA = cls._getOutputSchemaMapping(LLMChain)
        
        try:
            # Create a validator based on the JSON_SCHEMA
            validator = jsonschema.Draft7Validator(JSON_SCHEMA)
            
            # Check if the output is valid
            errors = list(validator.iter_errors(output))
            
            if errors:
                error_messages = [str(error) for error in errors]
                return (False, ", ".join(error_messages))  # Output is invalid, return the validation error messages
            else:
                return (True, "")  # Output is valid
        except Exception as e:
            return (False, str(e))  # Handle any other exceptions that may occur

class AgentMain:
    """Main Agent that Handles the LLM chaining of input and outputs for the Disruption Event Extraction"""
    binaryClassifier: LLMChain = articleClassifier
    locationExtractor: LLMChain = locationExtractor
    eventDetails: LLMChain = eventDetails 
    # The function to get the coordinates of a location
    get_coordinates: Callable = get_coordinates
    webscraper: Webscraper = webscraper
    article_token_threshold: int = 1500
    """The number of tokens to use the article summary instead of the full article"""
    @staticmethod
    def _articleExtraction(url) -> str:
        """Extracts the article text from the given url"""
        try:
            article = webscraper.scrape(url)
        except Exception as e:
            raise Exception(f"Error: Article Extraction Failed, {e}") from e
        return article
    
    @staticmethod
    def _binaryClassifier(article: Article) -> Union[dict, str]:
        """Classifies if the given article is a valid disruption event article or not
        Always use summary instead of full article for token saving
        """
        try:
            # Get number of tokens for the article
            result = AgentMain.binaryClassifier.run(articleTitle=article.title,articleText=article.summary, feedback="")

            # Validate the output
            validation_results = outputValidator.validate(AgentMain.binaryClassifier, result)
            if not validation_results[0]:
                logger.warning(f'_binaryClassifier Validation Error, Re-Running with feedback: {validation_results[1]}')
                # Re-run the validation error as feedback for the LLMChain
                result = AgentMain.binaryClassifier.run(articleTitle=article.title,articleText=article.text, feedback=validation_results[1])

        except Exception as e:
            raise Exception(f"Error: Binary Classification Failed -> {e}") from e
    
        return result
    @staticmethod
    def _locationExtractor(article: Article, feedback: str ="") -> str:
        """Extracts the location of the disruption event from the given article"""
        try:
            # Get number of tokens for the article
            num_tokens = num_tokens_from_string(article.text, "cl100k_base")
            if num_tokens > AgentMain.article_token_threshold:
                logger.info(f'Article Length: {num_tokens} tokens, using article summary instead')
                result = AgentMain.locationExtractor.run(articleTitle=article.title,articleText=article.summary, feedback=feedback)
            else:
                result = AgentMain.locationExtractor.run(articleTitle=article.title,articleText=article.text, feedback=feedback)
            # Validate the output
            validation_results = outputValidator.validate(AgentMain.locationExtractor, result)

            if not validation_results[0]:
                # Re-run the validation error as feedback for the LLMChain
                result = AgentMain.locationExtractor.run(articleTitle=article.title,articleText=article.text, feedback=validation_results[1])
                
        except Exception as e:
            raise Exception(f"Error: Location Extraction Failed -> {e}") from e
    
        logger.info(f'Successfully extracted disruption location: {result["location"]}')
        return result["location"]
    @staticmethod
    def _eventDetails(article: Article, feedback: str,force:bool = False) -> Union[dict, str]:
        """Extracts the event details of the disruption event from the given article"""
        try:
            # Get number of tokens for the article, if the article is too long, use the summary instead
            num_tokens = num_tokens_from_string(article.text, "cl100k_base")
            if num_tokens > AgentMain.article_token_threshold:
                logger.info(f'Article Length: {num_tokens} tokens, using article summary instead')
                result = AgentMain.eventDetails.run(articleTitle=article.title,articleText=article.summary, feedback=feedback)
            else:
                result = AgentMain.eventDetails.run(articleTitle=article.title,articleText=article.text, feedback=feedback)

            if not force:
                # Validate the output
                validation_results = outputValidator.validate(AgentMain.eventDetails, result)
                if not validation_results[0]:
                    # Return the validation error as feedback
                    return validation_results[1]
            
            
        except Exception as e:
            raise Exception(f"Error: Event Details Extraction Failed -> {e}") from e
        
        logger.info(f'Successfully extracted disruption Event Details: {result}')

        return result

    @staticmethod
    def articleAddParams(article: Article, params:dict) -> Article:
        """Adds the given params to Article.additional_data object"""
        # Add the params to the article dictionary
        article.additional_data.update(params)
        logger.info(f'Added number of params: {len(params)}')
        return article
    @classmethod
    def _process(cls, article: Article, url: str = None) -> Union[Article, str]:
        """Given an article or a url, use the LLMChains to extract the disruption event information
        Will return an Article object if the process is succesful
        Will return a string if the process failed
        Returns:
            Union[Article, str]: An Article object if the process is succesful, a string if the article non-disruption related/failed"""
        # try:
        if url:
            # Extract the article from the url as a newspaper3k Article object
            article = cls._articleExtraction(url)
            logger.info(f'Article Succesfully extracted from url: {article.title}')
        # Check if the article is a disruption event article
        logger.info(f'LLM Agent running STEP: Binary Classifier on article: {article.title}')
        classifier_result = cls._binaryClassifier(article)
        logger.info('classifier_result: ', classifier_result)

        if not classifier_result['isDisruptionEvent']:
            return f"NON-DISRUPTION EVENT ARTICLE -> {article.title},\nREASONING:{classifier_result['Reason']}"
        
        logger.info(f'SUCCESS: IS DISRUPTION EVENT ARTICLE -> {article.title}')
        logger.info(f'REASON FOR DETERMINING: {classifier_result["Reason"]}')
        # Extract the location of the disruption event
        location = cls._locationExtractor(article, feedback="")
        logger.info(f'Location extracted: {location}')
        coordinates_info = cls.get_coordinates(location)
        logger.info(f'Coordinates info: {coordinates_info}')
        # Loop until the coordinates are valid
        max_retries = 3
        retries = 0
        while isinstance(coordinates_info, str) and retries < max_retries:
            logger.warning(f'Location is not valid, please try again. Location: {location} coordinates: {coordinates_info}')
            location = cls._locationExtractor(article, feedback=coordinates_info)
            coordinates_info = cls.get_coordinates(location)
            retries += 1
        # Extract the disruption event information
        event_details = cls._eventDetails(article, feedback="")
        max_retries = 3
        retries = 0
        while isinstance(event_details, str) and retries < max_retries:
            logger.warning(f'Event Details is not valid, Running Agent again with feedback... -->{event_details}')
            # If last attempt, force the LLMChain to output whatever it has
            if retries == max_retries - 1:
                event_details = cls._eventDetails(article, feedback=event_details, force=True)
            else:
                event_details = cls._eventDetails(article, feedback=event_details)
            retries += 1
        logger.info(f'SUCCESS DETAILS EXTRACTED -> {event_details}')
        article = cls.articleAddParams(article, {"location": location, "coordinates": coordinates_info})
        article = cls.articleAddParams(article, event_details)
        article = cls.articleAddParams(article, classifier_result)

        return article

        # except Exception as e:
        #     logger.error(e)
        #     return str(e)
    @classmethod
    def processUrl(cls, url: str) -> Union[Article,str]:
        """Given a url, use the LLMChains to extract the disruption event information"""
        return cls._process(None, url)
    
    @classmethod
    def process(cls, article: Article) -> Union[Article,str]:
        """Main processing function, given an article, use the LLMChains to extract the disruption event information
        Returns article object if the process is succesful, a string if the article non-disruption related/failed"""
        return cls._process(article)