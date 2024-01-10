```python
import openai
from app.config import CONFIG
from app.utils import Tweet, Prediction

GPT3_MODEL = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
    ]
)

def analyze_sentiment(tweet: Tweet) -> Prediction:
    response = GPT3_MODEL.complete(
        prompt=f"{tweet.text}",
        max_tokens=60,
        temperature=0.4,
        top_p=1
    )

    sentiment_score = response['choices'][0]['finish_reason']
    magnitude = response['choices'][0]['logprobs']['token_logprobs'][0]

    if sentiment_score > 0:
        sentiment = "POSITIVE"
    else:
        sentiment = "NEGATIVE"

    return Prediction(tweet.id, sentiment, magnitude)
```