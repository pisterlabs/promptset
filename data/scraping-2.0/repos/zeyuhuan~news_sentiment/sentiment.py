from openai import OpenAI
import json

client = OpenAI()


def append_sentiment(news: list[dict]) -> list[dict]:
    user_messages = [
        {
            "role": "user",
            "content": f"Name: {individual_news['Name']}, Description: {individual_news['Description']}",
        }
        for individual_news in news
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": 'You are a service to analyze sentiment. \
                    Respond strictly in such format: [{"Sentiment": <your response>, "Reason": <your response>}, {"Sentiment": <your response>, "Reason": <your response>}, ...]. \
                        Make sure the length of response list is the same as size as input',
            }
        ]
        + user_messages,
    )

    sentiments = json.loads(response.choices[0].message.content)

    assert len(sentiments) == len(news)

    for individual_news, sentiment in zip(news, sentiments):
        individual_news["Sentiment"] = sentiment["Sentiment"]
        individual_news["Reason"] = sentiment["Reason"]

    return news
