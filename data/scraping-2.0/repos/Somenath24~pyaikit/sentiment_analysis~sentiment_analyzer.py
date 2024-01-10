# Import the required libraries
import requests
import PySimpleGUI as sg
import os
import openai
import pandas as pd

# Define a class for sentiment analysis
class sentiment_analyzer:
    # Define the instance variables for the class
    def __init__(self):
        self.messages = ""
    
    # Run the OpenAI sentement analysis model
    def generate_basic_sentiment(self, openai, messages):
        """
        Generate basic sentiment using OpenAI text completion.

        Args:
            self (object): The instance of the class.
            openai (object): The OpenAI object for accessing the API.
            messages (str): The input text for sentiment analysis.

        Returns:
            str: The basic sentiment as a string indicating positive, negative, or neutral sentiment.
                Returns 'No responses' if no sentiment is detected.

        """

        # Create the prompt for basic sentiment analysis
        prompt = "Sentiment analysis of the following text, give only if it is positive, negative, or neutral: " + messages

        # Generate basic sentiment using OpenAI text completion
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # Process the response and extract the basic sentiment
        if 'choices' in response:
            if len(response['choices']) > 0:
                ret = response['choices'][0]['text']
                ret = ret.replace("\n", "")
                ret = ret.strip()
            else:
                ret = 'No responses'
        else:
            ret = 'No responses'

        return ret

    
    # Run the OpenAI sentement analysis model
    def generate_advanced_sentiment(self, openai, messages):
        """
        Generate advanced sentiment using OpenAI text completion.

        Args:
            self (object): The instance of the class.
            openai (object): The OpenAI object for accessing the API.
            messages (str): The input text for sentiment analysis.

        Returns:
            str: The advanced sentiment as a string indicating emotions such as happy, sad, anger, irritated, or surprise.
                Returns 'No responses' if no sentiment is detected.

        """

        # Create the prompt for advanced sentiment analysis
        prompt = "Sentiment analysis of the following text, give only if it is either happy, sad, anger, irritated or surprise: " + messages

        # Generate advanced sentiment using OpenAI text completion
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # Process the response and extract the advanced sentiment
        if 'choices' in response:
            if len(response['choices']) > 0:
                ret = response['choices'][0]['text']
                ret = ret.replace("\n", "")
                ret = ret.strip()
            else:
                ret = 'No responses'
        else:
            ret = 'No responses'

        return ret

    
    def generate_sentiment_score(self, openai, messages):
        """
        Generate sentiment score using OpenAI text completion.

        Args:
            self (object): The instance of the class.
            openai (object): The OpenAI object for accessing the API.
            messages (str): The input text for sentiment analysis.

        Returns:
            str: The sentiment score as a numerical value between -1 and 1, indicating sentiment. 
                0 represents neutral sentiment, +1 indicates positive sentiment, and -1 indicates negative sentiment.

        """

        # Create the prompt for sentiment analysis
        prompt = "\"Please provide the sentiment in a numerical score between -1 and 1. The score should indicate the sentiment, with 0 being neutral, +1 indicating a positive sentiment, and -1 indicating a negative sentiment, now the text will come in single quotes '" + messages + "' \""

        # Generate sentiment score using OpenAI text completion
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=1,
            max_tokens=10,
            top_p=1,
            frequency_penalty=0.5,
            presence_penalty=0.5
        )

        # Process the response and extract the sentiment score
        if 'choices' in response:
            if len(response['choices']) > 0:
                ret = response.choices[0].text
                ret = ret.replace("\n", "")
                ret = ret.strip()
            else:
                ret = 'No responses'
        else:
            ret = 'No responses'

        return ret

    
    def generate_sentiment_mult(self, openai, list_of_text=None, sent_type="score"):
        """
        Generate sentiment for multiple texts using OpenAI sentiment models.
        Args:
            self (object): The instance of the class.
            openai (object): The OpenAI object for accessing sentiment models.
            list_of_text (list, optional): A list of texts to analyze. Defaults to None.
            sent_type (str, optional): The type of sentiment analysis to perform. Possible values are "score" (default),
                                    "advanced", and "basic".

        Returns:
            list: A list containing the sentiment analysis results for each text.

        Raises:
            None.

        """

        if list_of_text is None:
            print("List is missing, pass a list of text")
        else:
            output = []
            for text in list_of_text:
                if sent_type == "score":
                    # Perform sentiment analysis using OpenAI sentiment score model
                    ret = self.generate_sentiment_score(openai, text)
                elif sent_type == "advanced":
                    # Perform advanced sentiment analysis using OpenAI advanced sentiment model
                    ret = self.generate_advanced_sentiment(openai, text)
                else:
                    # Perform basic sentiment analysis using OpenAI basic sentiment model
                    ret = self.generate_basic_sentiment(openai, text)
                
                output.append(ret)

        return output