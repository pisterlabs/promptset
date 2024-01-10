import openai
from recommender import config
from recommender import templates
from easydict import EasyDict as edict

openai.api_key = config.OPENAI_API_KEY
sentiments = ["happy", "sad", "excited", "angry", "neutral"]
answers = ["We are glad that this recommendation made you happy, we will try to recommend you more things like this.",
            "While being sad is okay, we will try to bring more happy memories.",
            "Remembering es exciting! Thanks for letting us know what you think.",
            "We see that this topic is not of your preference, we will try to recommend less things like this.",
            "Maybe this recommendation was not very relevant to you, we will try to show you a larger variety of recommendations."]
def analyze_sentiment(text,engine="text-davinci-002"):
    response = openai.Completion.create(
    engine=engine,
    prompt=templates.feedback.format(text),
    temperature=0.5,
    max_tokens=16,
    top_p=1.0,
    frequency_penalty=0,
    presence_penalty=0
    )
    response = response["choices"][0]["text"]
    print(response)
    index = 0
    for (i,s) in enumerate(sentiments):
        if s in response:
            index = i
    print(index)
    print(answers[index])
    return answers[index]