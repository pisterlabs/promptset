import os
import openai
import json
import random
import glob
import tqdm

import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import pandas as pd

random.seed(71)

def generate_blurb_addidas(article):
    blurb = "List out all the names and stock tickers of the companies most associated with the news, and how the price will move as a result of the news:\n\n"
    blurb += "TITLE: Adidas Frees the Nipple With New Sports-Bra Campaign for Women\n"
    blurb += "SUMMARY: The German sportswear brand released an internet-breaking image of bare breasts to accompany a sports-bra collection of 43 diverse styles. Adidas launched its new sports-bra campaign today with an unusual image: a grid of 25 sets of bare breasts. Aside from the surprise factor of the nudity, the breasts shown are only remarkable because they are completely normal. These are real breasts in all their perky, saggy, asymmetrical, varied forms."
    blurb += "\n\n- Adidas (ADS):  Adidas' stock price is likely to go up because of the positive publicity the company is receiving for its new campaign.\n- Nike (NKE): Nike's stock price is likely to go down because of the competition from Adidas.\n\n"


    blurb += "List out all the names and stock tickers of the companies most associated with the news, and how the price will move as a result of the news:\n\n"
    blurb += "TITLE: " + article["title"]
    blurb += "\nSUMMARY: " + article["summary"]

    return blurb


def generate_blurb(article, base):
    #read from data/prompts/flight_prompt.txt
    with open(base) as f:
        prompt = f.read()
        prompt = prompt.replace("{TITLE}", article['title'])
        prompt = prompt.replace("{SUMMARY}", article['summary'])
        return prompt

nyse = mcal.get_calendar('NYSE')
schedule = nyse.schedule(start_date='2019-01-01', end_date="2022-03-01")
#Check if the NYSE Market is open at the given date
def valid_date(date):
    try:
        return nyse.open_at_time(schedule, pd.Timestamp(date, tz='America/New_York'))

    except:
        return False


def keep_article(article):
    if not valid_date(article['date']):
        return False

    category_keywords = ["Business", "WSJ News Exclusive", "Markets", "Heard on the Street"] # ['Business', 'Economy', 'Heard on the Street', 'Markets', 'Tech', 'Stocks', 'WSJ News Exclusive', 'CFO Journal', 'Pro Bankruptcy', 'CIO Journal']
    #not included: opinion, life & work, World, US, Asia
    #Old keywords: keywords = ['Business', 'Economy', 'Heard on the Street', 'Markets', 'Tech', 'Stocks', 'Opinion', 'Life & Work']

    banned_categories = ['Journal Reports: Small Business']
    category_urls = ['https://www.wsj.com/news/types/deals-deal-makers?mod=breadcrumb',
                        'https://www.wsj.com/news/heard-on-the-street?mod=breadcrumb',
                        'https://www.wsj.com/news/types/today-s-markets?mod=breadcrumb',
                        'https://www.wsj.com/news/markets?mod=breadcrumb',
                        'https://www.wsj.com/news/business?mod=breadcrumb',
                        'https://www.wsj.com/news/life-work/automotive?mod=breadcrumb',
                        'https://www.wsj.com/news/types/rumble-seat?mod=breadcrumb',
                        'https://www.wsj.com/news/types/technology?mod=bigtop-breadcrumb',
                        'https://www.wsj.com/news/types/cfo-journal?mod=breadcrumb']
    
    if 'categories' in article:
        if any([ck in article['categories'][0] for ck in banned_categories]):
            return False
        if any([ck in article['categories'][0] for ck in category_keywords]):
            return True
    if 'category_urls' in article:
        if any([cu in article['category_urls'] for cu in category_urls]):
            return True
    return False


def call_openai(blurb):
    
    openai.Engine.retrieve("text-davinci-001")
    response = openai.Completion.create(
            engine="text-davinci-001",
            prompt=blurb,
            max_tokens=96, #previously 64, 0
            temperature=.21)

    return response

def init_openai():
    openai.organization = 'org-j6stjF9ItWk2DOuv9iDZ0lxH'
    openai.api_key = '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'
    openai.Engine.list()


def main():
    init_openai()

    news_files = glob.glob("data/wsj/*.json")
    #randomly sample from the files
    # news_files = random.sample(news_files, 50)
    cnt = 0

    result_objs = []

    for filename in tqdm.tqdm(news_files):
        #read the file as json
        with open(filename) as f:
            data = json.load(f)

            #For each article in data, print whether it is a valid article using keep_article
            # for article in data:
            #     if not keep_article(article):
            #         print()
            #         print(article["title"])
            #         print(article['categories'][0] if 'categories' in article else "")
            #filter the articles using keep_article
            articles = list(filter(keep_article, data))
            cnt += len(articles)
            for article in articles:
                if random.random() >= 0:
                    continue
                blurb = generate_blurb(article, "data/prompts/vaccine_prompt.txt")
                
                article['blurb'] = blurb
                response = call_openai(blurb)
                article['Prediction'] = response['choices'][0]['text']

                result_objs.append(article)
                len(result_objs)
                # save checkpoints
                if len(result_objs) % 5 == 0:
                    print("Saving checkpoint...")
                    with open("data/wsj_predictions/vaccine_prompt_filtered.json", "w") as f:
                        json.dump(result_objs, f)
    print(cnt)

    #save result_objs to a file

    with open("data/wsj_predictions/vaccine_prompt_filtered.json", "w") as f:
        json.dump(result_objs, f)
    

if __name__ == '__main__':
    main()
