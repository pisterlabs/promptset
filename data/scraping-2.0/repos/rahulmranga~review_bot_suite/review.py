from pydantic import BaseModel
#import constants
import openai
from transformers import pipeline
import torch
from review_aspects import get_topics,get_sentiments



#key =ENTER_API_KEY

openai.api_key = ''

class ReviewAttr(BaseModel):
    def __init__(self):
        review_text: str
        company_name: str
        self.sentiment_pipeline = pipeline("sentiment-analysis")

    def get_review_sentiment(self,review_text: str) -> str:
        
        response=self.sentiment_pipeline(review_text)
        if response['label']=='POSITIVE':
            return 'Positive'
        elif response['label']=='NEGATIVE':
            return 'Negative'
        else:
            return 'Neutral'
        

    def get_topics(self,review: str,source:str) -> str:
        if(source=='GPT'):
            prompt = "What did the customer like/dislike in this review? Respond with the topic of what was liked/disliked in only a single word or two. If the customer liked/disliked about more than one topic, separate the topics with a comma and its sentiment: "

            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=.2,
            messages=[
                    {"role": "user", "content": f"{prompt}{review}"}
                ]
            )

            topics_sentiment = str(response['choices'][0]['message']['content'])
        else:
            topics=get_topics(review)
            topics_sentiment=get_sentiments(review,topics)
        return topics_sentiment