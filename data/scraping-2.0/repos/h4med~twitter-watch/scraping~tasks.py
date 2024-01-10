import asyncio
from parsel import Selector
from playwright.async_api import async_playwright
from datetime import datetime, timedelta
from celery import shared_task
from .models import Tweet
from asgiref.sync import sync_to_async
import openai
from decouple import config

openai.api_key = config('OPENAI_API_KEY')

@sync_to_async
def get_tweet_by_id(id):
    tweet = Tweet.objects.filter(tweet_id=id)
    return tweet

@sync_to_async
def save_item_to_db(id, data):

    if 'image' not in data:
        data['image'] = ''
    if 'video' not in data:
        data['video'] = ''
    if 'username' not in data:
        data['username'] = ''

    user= data['handle']
    try:
        Tweet.objects.update_or_create(
            tweet_id = id,
            publish_date = data['datetime'],
            defaults={
                'user_name' : data['username'],
                'image_url' : data['image'],
                'video_url' : data['video'],
                'text' : data['text'],
                'handle' : data['handle'],
                'likes' : data['likes'],
                'retweets' : data['retweets'],
                'replies' : data['replies'],
                'views' : data['views'],            
            }
        )

        print(f"item {id} created/updated for {user}")

    except Exception as e:
        print(f"failed at create/update item {id} for {user}")
        print(e)

async def save_data(data, account):
    print('in save_data function')
    # print(data)

    for id in data.keys():
        try:
            tweet = await get_tweet_by_id(id)
            print(tweet)
        except Exception as e:
            user = data[id]["handle"]
            print(f'Exception at finding tweet by id: {id} from {user}')
            print(e)
            pass
        finally:
            await save_item_to_db(id, data[id])
        
def parse_tweets(selector: Selector):
    results = []

    tweets = selector.xpath("//article[@data-testid='tweet']")
    for tweet in tweets:
        found = {
            "text": "".join(tweet.xpath(".//*[@data-testid='tweetText']//text()").getall()),
            "username": tweet.xpath(".//*[@data-testid='User-Names']/div[1]//text()").get(),
            "handle": tweet.xpath(".//*[@data-testid='User-Names']/div[2]//text()").get(),
            "datetime": tweet.xpath(".//time/@datetime").get(),
            "url": tweet.xpath(".//time/../@href").get(),
            "image": tweet.xpath(".//*[@data-testid='tweetPhoto']/img/@src").get(),
            "video": tweet.xpath(".//video/@src").get(),
            "likes": tweet.xpath(".//*[@data-testid='like']//text()").get(),
            "retweets": tweet.xpath(".//*[@data-testid='retweet']//text()").get(),
            "replies": tweet.xpath(".//*[@data-testid='reply']//text()").get(),
            "views": (tweet.xpath(".//*[contains(@aria-label,'Views')]").re("(\d+) Views") or [None])[0],
        }
        results.append({k: v for k, v in found.items() if v is not None})

    return results


async def run(playwright):
    accounts = ["BarackObama", "CathieDWood", "elonmusk"]
    # accounts = ["BarackObama"]
    final_id_val_data = {}
    try:
        for account in accounts:
            print(f"Scraping data for {account}\n")

            chromium = playwright.chromium # or "firefox" or "webkit".
            browser = await chromium.launch() # default, when using proxy use following settings
            # browser = await chromium.launch(proxy={
            # "server": "socks5://127.0.0.1:10808",
            # })

            page = await browser.new_page()

            thirthDaysAgo = (datetime.now() - timedelta(30)).isoformat()
            # Feb1st = datetime(2023,2,1,0,0,0,0).isoformat()
            # Feb1st = datetime(2023,3,5,0,0,0,0).isoformat() # for shorter tests


            await page.goto("https://twitter.com/"+account)
            raw_data = []

            datetime_var = datetime.now().isoformat()
            page_scroll = 0

            while datetime_var > thirthDaysAgo:
                await page.evaluate(f"window.scrollBy(0, {page_scroll * 720})")
                await page.wait_for_selector("//article[@data-testid='tweet']") 
                html = await page.content()
                # parse it for data:
                selector = Selector(html)
                tweets = parse_tweets(selector)
                last_date_scrapped = tweets[-1]['datetime']
                print(f"page: {page_scroll} -- {last_date_scrapped}")
                # print(tweets)

                i = 0
                for anytw in tweets:
                    if 'handle' not in anytw:
                        tweets[i]['handle'] = "@"+account
                    i +=1

                raw_data.extend(tweets)

                datetime_var =  tweets[-1]['datetime']
                page_scroll += 1

            print(f"Reached Feb 1st 2023, after {page_scroll} pages of scrolling")

            for tw in raw_data:
                # print(tw)
                # print("-------------------------------")
                if tw['handle'] == "@"+account:
                    key = tw['url'].split("/")[-1]
                    final_id_val_data[key] = tw

            await browser.close()

            print("Next step: save function\n")
            save = save_data(final_id_val_data, account)
        return await save

    except Exception as e:
        print('____The scraping job for failed. See exception:')
        print(e)  

async def main():
    async with async_playwright() as playwright:
        await run(playwright)

@shared_task
def scraping_method():
    asyncio.run(main())

def sentiment_detection_openai(tweet):
    the_prompt = "Classify the sentiment in this tweet:\n\n " + tweet + "\n\n"
    # print(the_prompt)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=the_prompt,
        temperature=0,
        max_tokens=300,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    return response['choices'][0]['text'].replace("\n", "").strip()

@shared_task
def sentiment_detection():
    print('_____________ sentimetn_detection Starts Running _____________')
    queryset = Tweet.objects.all()
    cntr = 0
    for tweet in queryset:
        if tweet.sentiment == '' and tweet.text != '':
            tweet.sentiment = sentiment_detection_openai(tweet.text)
            tweet.save()
            print(f'{tweet.sentiment} saved for tweet: {tweet.tweet_id} by {tweet.handle}')

            cntr += 1
            if cntr > 25:
                print('Reached OpenAI limit rate, BREAK!')
                break # openAi Limit rate
    if cntr == 0:
        print('No Tweet without sentiment OR text, to detect the sentiment!')