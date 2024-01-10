import re
import openai
import matplotlib.pyplot as plt
from collections import defaultdict
from decouple import config

OPENAI_API_KEY = config("OPENAI_API_KEY")


def analyze_subtitle2(file_path):
    with open(file_path, "r") as file:
        text = file.read()
        blocks = re.findall(
            r"(\d{2}:\d{2}:\d{2},\d+) --> (\d{2}:\d{2}:\d{2},\d+)\n(.*?)\n\n",
            text,
            re.DOTALL,
        )

        text_by_minute = defaultdict(str)

        for start, end, text in blocks:
            hour, minute, second, _ = map(int, re.split("[:,]", start))
            total_minutes = (
                hour * 60 + minute
            )  # converting hours to minutes and adding to the minute part
            clean_text = re.sub(r"\W+", " ", text).replace("\n", " ").strip()
            text_by_minute[total_minutes] += " " + clean_text

        avg_sentiments = []
        sentiment_sum = 0
        count = 0

        for idx, (minute, text) in enumerate(sorted(text_by_minute.items())):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f'Sentiment analysis of the text from minute {minute} to {minute}: "{text}". Provide a sentiment score between 0 and 10. Only a single number! No text or explanation needed, just a number.',
                    },
                ],
                api_key=OPENAI_API_KEY,
            )
            sentiment_score = float(
                response["choices"][0]["message"]["content"].strip()
            )
            print(f"Sentiment score for minute {minute}: {sentiment_score}")
            sentiment_sum += sentiment_score
            count += 1

            if (idx + 1) % 5 == 0:
                avg_sentiments.append(sentiment_sum / count)
                sentiment_sum = 0
                count = 0

        if count > 0:
            avg_sentiments.append(sentiment_sum / count)

        return avg_sentiments
