import openai
from .ChatModelCompletor import get_completion_from_messages as get_completion_from_messages

class CommentSentimentAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def analyze_sentiment(self, customer_comment):
        delimiter = "####"

        system_message = f"""
        You are a customer intention analyzer of the store.\
        Analyze the sentiment of the following customer comment, tell the colleagues that is it Positive or Negative.\
        The customer comment will be delimited with \
        {delimiter} characters.\
        Only reply Positive Or Negative, with nothing else.\
        """

        user_message_1 = f"""
            {customer_comment} \
        """

        messages =  [  
            {'role':'system', 
            'content': system_message},    
            {'role':'user', 
            'content': f"{delimiter}{user_message_1}{delimiter}"},  
        ] 
        
        return get_completion_from_messages(messages)