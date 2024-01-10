from typing import Tuple
import openai
import json
import os
import logging
from textblob import TextBlob

# Initialize logging
logging.basicConfig(level=logging.INFO)

def analyze_sentiment(comment: str) -> Tuple[float, str, str]:
    try:
        openai.api_key = os.environ.get('API_KEY')
        prompt_text = f'''make a sentiment analysis on following comment:
        "{comment}"

        Return results as json, with following attributes
        "polarity_score",
        "sentiment",
        "explanation"
        '''
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "assistant", "content": prompt_text}])
        content = completion.choices[0].message.content
        if not content.endswith('}'):
            content += '}'        
        result = json.loads(content)        
        return result['polarity_score'], result['sentiment'].lower(), result['explanation']
    except Exception as e:
        logging.warning(f"Falling back to TextBlob due to an error: {str(e)}")
        analysis = TextBlob(comment)
        polarity = analysis.sentiment.polarity
        classification = "positive" if polarity > 0 else "negative"
        explanation = ""  # Initialize an empty string for explanation
        return polarity, classification, explanation