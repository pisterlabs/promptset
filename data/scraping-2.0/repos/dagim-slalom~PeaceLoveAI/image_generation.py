from openai import OpenAI

client = OpenAI()


def qualify_sentiments(sentiment_value):
    if sentiment_value <= 1:
        return "very sad"
    if sentiment_value <= 2:
        return "sad"
    if sentiment_value <= 3:
        return "neutral"
    if sentiment_value <= 4:
        return "happy"
    if sentiment_value <= 5:
        return "very happy"


def lambda_handler(event):
    initial_sentiment = qualify_sentiments(event["pre_game_average"])
    final_sentiment = qualify_sentiments(event["post_game_average"])
    game_sentiment = qualify_sentiments(event["session_average"])

    session_prompt = f"""Depict an abstract artistic representation of a heatmap, with gradients of color intensity
         representing a range of sentiments. The initial sentiment is {initial_sentiment}.
         Then a sentiment of {game_sentiment} and ends with a sentiment of {final_sentiment}.
         There should be no text or numbers"""
    response = client.images.generate(
        model="dall-e-3",
        prompt=session_prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url

    print(image_url)

    return { "image": image_url,
             "sessionId": event["sessionId"]
    }

# # Test Value
# lambda_handler({"pre_game_average": 2.1, "post_game_average": 4.1,
#                 "session_average": 3.7, "sessionId": "2789"})
