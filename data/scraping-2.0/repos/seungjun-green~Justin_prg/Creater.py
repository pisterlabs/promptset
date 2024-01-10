import random
import openai
from Resources import data
from datetime import datetime
from Brain import Brain
from Twitter import Twitter
import Settings
import re

class Creater:
    def send_tweet(self, order):
        # creating content
        result = ""
        print(f"the order is:\n {order}")
        try:
            #create and process the string
            result = Brain().create_content(order)
            result = self.process_str(result)
            print(f"new tweet is {result}")
        except openai.error.OpenAIError as e:
            print(f"[send_tweet] openAI Error: {e}\n")

        # tweeting the content
        if Settings.production:
            Twitter().tweet_content(result)
        else:
            print("content tweeted - Development mode\n")

    def get_type(self):
        if Settings.production:
            now = datetime.now().time().replace(second=0, microsecond=0)
            currH, currM = now.hour, now.minute
            curr = (currH, currM)
            if curr in data.news_times:
                return 'news'
            else:
                return None
        else:
            return "news"

    def process_str(self, result):
        result = re.sub('@[a-zA-Z_0-9]*', '', result)
        result = re.sub('#[a-zA-Z_0-9]*', '', result)

        return result

    def process_data(self, data):
        scoreboard = []

        for i in range(0, len(data)):
            curr_raw = data[i]._json
            curr = {}
            curr['tweet_id'] = curr_raw['id']
            if 'retweeted_status' in curr_raw:
                raw_text = curr_raw['retweeted_status']['full_text']
            else:
                raw_text = curr_raw['full_text']

            processed_text = re.sub(r'RT', "", raw_text)
            processed_text = re.sub(r'@\w+:', "", processed_text)
            processed_text = re.sub(r'@\w+', "", processed_text)
            processed_text = re.sub(r'@:', "", processed_text)
            processed_text = re.sub(r'@', "", processed_text)
            processed_text = re.sub(r'https://t.co/\w+', "", processed_text)
            processed_text = re.sub(r'#\w+', "", processed_text)
            processed_text = re.sub(r'#', "", processed_text)
            processed_text = re.sub(r'\n', "", processed_text)
            processed_text = re.sub(' +', ' ', processed_text)
            processed_text = re.sub(r'\[Feature]', "", processed_text)
            curr['text'] = processed_text.strip()
            curr['score'] = 0
            print(curr)

            scoreboard.append(curr)

        return scoreboard

    def create_order(self):

        import datetime
        today = datetime.date.today()

        if today.day % 2 == 0:
            word = random.choice(['SwiftUI', 'UIKit'])
        else:
            word = random.choice(['Python', 'Machine Learning'])

        pormpt = ""
        fetched_tweets = set()

        if Settings.production:
            raw_data = Twitter().fetch_tweets(word)
            cleaned_data = self.process_data(raw_data)
        else:
            cleaned_data = Settings.example_data

        print("Did we came here?")
        print(len(cleaned_data))
        for row in cleaned_data:
            if '100' not in row['text'] and row['text'] not in fetched_tweets:
                curr = f"tweet: {row['text']}\n"
                fetched_tweets.add(row['text'])
                pormpt += curr
                print(f"** {pormpt}")
            else:
                pass

        pormpt += '\n  using above content, create your own tweet as if you experienced one of above:'
        return pormpt