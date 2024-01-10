import praw
from IPython.core.display import display
from yahoo_fin.stock_info import get_data
from datetime import date, timedelta
import datetime as dt
import requests
from psaw import PushshiftAPI
import json
import os
import openai
import re
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt

# 'link_flair_text': ':DDNerd: DD :DD:',

# regex for ticker symbols, will be used when searching subreddit titles
regex_for_ticker = "\$[A-Z]{3}[A-Z]?[A-Z]?"

# openai key, delete before pushes
openai.api_key = "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"

# list of tickers and if they have supporting posts
# stock_list = []

ticker_list = []

has_multiple_posts = []

time_to_sell_list = []

# start and end date for searches
start_epoch = int(dt.datetime(2022, 11, 27).timestamp())
end_epoch = int(dt.datetime(2022, 12, 9).timestamp())
start_date = date(2022, 11, 27)
end_date = date(2022, 12, 9)

# reddit connection and connection to Pushshift api
reddit_read_only = praw.Reddit(client_id="jIecQpWnOYUzYOgTmdEsTg",  # your client id
                               client_secret="MF8q2iDGYcsoKY3mta49Lu5frWa5MQ",  # your client secret
                               user_agent="MyBot/0.0.1")  # your user agent
api = PushshiftAPI(reddit_read_only)

sobr_daily = get_data("sobr", start_date="11/27/2022", end_date="12/09/2022", index_as_date=True, interval="1d")
sobr_daily2 = get_data("sobr", start_date="12/01/2022", end_date="12/02/2022", index_as_date=True, interval="1d")
sobr_daily3 = get_data("sobr", start_date="12/06/2022", end_date="12/07/2022", index_as_date=True, interval="1d")

money = 10000.0
standard_num_shares = 100

# generates submission objects where we can get data from (ex. submission.title)
# myListExample = api.search_submissions(after=start_epoch,
#                                        before=end_epoch,
#                                        subreddit='pennystocks',
#                                        filter=['url', 'author', 'title', 'subreddit'],
#                                        limit=10,
#                                        # score='>10',
#                                        title='$SOBR')
# will show all variables in the objects returned from myList
# for x in myListExample:
#     pprint(vars(x))

def daterange(s_d, e_d):
    for n in range(int((e_d - s_d).days)):
        yield s_d + timedelta(n)


for single_date in daterange(start_date, end_date):
    # print(single_date.strftime("%Y-%m-%d"))
    one_day_later = single_date + dt.timedelta(days=1)
    temp_start_epoch = int((dt.datetime.combine(single_date, dt.datetime.min.time())).timestamp())
    temp_end_epoch = int((dt.datetime.combine(one_day_later, dt.datetime.min.time())).timestamp())
    for num_days_left in time_to_sell_list:
        num_days_left -= 1
        print("NUMBER DAYS LEFT: " + str(num_days_left))
        if num_days_left == 0:
            money += 100 * sobr_daily3.open[0]

    print(
        single_date.strftime("%Y-%m-%d, %H:%M:%S") + " one day later: " + one_day_later.strftime("%Y-%m-%d, %H:%M:%S"))
    myList = api.search_submissions(after=temp_start_epoch,
                                    before=temp_end_epoch,
                                    subreddit='pennystocks',
                                    filter=['url', 'author', 'title', 'subreddit'],
                                    limit=10,
                                    # score='>10',
                                    title='$SOBR')

    for submission in myList:
        temp_ticker = re.findall(regex_for_ticker, submission.title)[0]
        print("Title: " + submission.title + " Body: " + submission.selftext + " Score: " + str(
            submission.score) + " Url: " + submission.url)
        responseTitle = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Classify the sentiment in this: {submission.title}",
            temperature=0,
            max_tokens=60,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        print("Sentiment of the title is: " + responseTitle["choices"][0]["text"].replace("\n", ""))
        responseBody = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Classify the sentiment in this: {submission.selftext}",
            temperature=0,
            max_tokens=60,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        print("Sentiment of the body is: " + responseBody["choices"][0]["text"].replace("\n", ""))
        if submission.link_flair_text == ":DDNerd: DD :DD:":
            if temp_ticker not in ticker_list:
                if submission.score >= 5 and (responseTitle["choices"][0]["text"].replace("\n", "") == "Positive" or \
                        responseBody["choices"][0]["text"].replace("\n", "") == "Positive"):
                    ticker_list.append(temp_ticker)
                    has_multiple_posts.append(False)
                    time_to_sell_list.append(-1)
                    print("SUBMISSION FOUND")
                    print(temp_ticker)
                    print("Score: " + str(submission.score))
            else:
                if has_multiple_posts[ticker_list.index(temp_ticker)] is False and (responseTitle["choices"][0][
                    "text"].replace("\n", "") == "Positive" or responseBody["choices"][0]["text"].replace("\n",
                                                                                                          "") == "Positive"):
                    has_multiple_posts[ticker_list.index(temp_ticker)] = True
                    print("The stock: " + temp_ticker + " has been backed : " + str(
                        has_multiple_posts[ticker_list.index(temp_ticker)]))
                    # buy the stock
                    money -= standard_num_shares * sobr_daily2.open[0]
                    print("money : " + str(money))
                    time_to_sell_list[ticker_list.index(temp_ticker)] = 5


for i in ticker_list:
    print(str(i))

# sobr_daily = get_data("sobr", start_date="11/27/2022", end_date="12/09/2022", index_as_date=True, interval="1d")
# pprint(vars(sobr_daily))
display(sobr_daily2)
print("Ending money: " + str(money))

# sobr_daily.plot()

# plt.show()


# f = open("wasd.txt", "a")
# f.write(x)
# f.close()

# display(myList[0])

# query = "pennystocks"  # Define Your Query
# query2 = ">10"
# url = f"https://api.pushshift.io/reddit/search/submission/?subreddit={query}&?score={query2}"
# request = requests.get(url)
# json_response = request.json()

# Serializing json
# json_object = json.dumps(json_response, indent=4)

# Writing to sample.json
# with open("sample.json", "w") as outfile:
#     outfile.write(json_object)

# f = open("demofile2.txt", "a")
# f.write(json_response)
# f.close()
# display(json_response['data'])

# subreddit = reddit_read_only.subreddit("pennystocks")

# Display the name of the Subreddit
# print("Display Name:", subreddit.display_name)
#
# # Display the title of the Subreddit
# print("Title:", subreddit.title)
#
# # Display the description of the Subreddit
# print("Description:", subreddit.description)

# posts = subreddit.top("month")
# Scraping the top posts of the current month

# posts_dict = {"Title": [], "Post Text": [],
#               "ID": [], "Score": [],
#               "Total Comments": [], "Post URL": []
#               }
#
# for post in posts:
#     # Title of each post
#     posts_dict["Title"].append(post.title)
#
#     # Text inside a post
#     posts_dict["Post Text"].append(post.selftext)
#
#     # Unique ID of each post
#     posts_dict["ID"].append(post.id)
#
#     # The score of a post
#     posts_dict["Score"].append(post.score)
#
#     # Total number of comments inside the post
#     posts_dict["Total Comments"].append(post.num_comments)
#
#     # URL of each post
#     posts_dict["Post URL"].append(post.url)

# top_posts = pd.DataFrame(posts_dict)
# print(tabulate(top_posts))

# display(top_posts)
