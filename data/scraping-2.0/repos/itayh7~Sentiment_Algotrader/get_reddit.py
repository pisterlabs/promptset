import openai
from SecretKeys import *
import praw
from tqdm import tqdm
from pytrends.request import TrendReq
import pandas as pd
import time
import requests
from datetime import datetime, timedelta

class RedditPost:
    def __init__(self, title: str, ups: int, downs: int, num_comments: int, date: datetime):
        self.title = title
        self.ups = ups
        self.downs = downs
        self.num_comments = num_comments
        self.sentiment=-1
        self.date = date

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_SECRET_KEY,
    user_agent="scraper 1.0 algo-project",
)

NUM_OF_STOCKS = 20
SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "stockmarket",
    "trading",
    "forex",
    "investor",
    "technicalraptor",
    "tradingreligion",
    "asktrading",
    "etoro",
    "bulltrader",
    "iama",
    "finance",
    "forextrading",
    "crowdfunding",
    "stocktrader",
    "wealthify",
    "investoradvice",
]
SUBREDDIT_POST_LIMIT = 10000


def analyze_sentiment(text: str) -> int:
    openai.api_key = OPENAI_API_KEY

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "You are a sentiment claissifier of the following text on a scale of 1 to 10. respond only the number",
                },
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=6,
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.0,
        )
        answer = response.choices[0].message.content
        return (
            int(answer)
            if answer.isdigit() and int(answer) > 0 and int(answer) < 11
            else 5
        )
    except openai.error.RateLimitError:
        print("Rate limit exceeded. Retrying in 5 seconds...")
        time.sleep(5)
        return analyze_sentiment(text)


def get_subreddits_posts_praw() -> list[RedditPost]:
    posts = []
    # print('Getting posts from r/'+subreddit+'...')
    for subreddit in SUBREDDITS:
        try:
            for post in tqdm(
                reddit.subreddit(subreddit).new(limit=SUBREDDIT_POST_LIMIT),
                desc=f"Fetching posts from {subreddit}",
            ):
                # check if post is a text post and not a meme, and not a daily discussion thread
                if (
                    post.link_flair_text != "Meme"
                    and "Daily Discussion Thread" not in post.title
                ):
                    # encode-decode will remove emojis from the title
                    title = post.title.encode("ascii", "ignore").decode()
                    posts.append(
                        RedditPost(title, post.ups, post.downs, post.num_comments, datetime.utcfromtimestamp(post.created_utc)))
      
            print(f"Posts from {subreddit} fetched successfully!\n")
        except:
            print(f"Failed to fetch posts from {subreddit}!")
    return posts


def get_reddit_stock_posts(stock: str) -> list:
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_SECRET_KEY,
        user_agent="scraper 1.0 algo-project",
    )
    posts = []
    for post in reddit.subreddit("all").search(stock, limit=SUBREDDIT_POST_LIMIT):
        # check if post is a text post and not a meme, and not a daily discussion thread
        if (
            post.link_flair_text != "Meme"
            and "Daily Discussion Thread" not in post.title
        ):
            # encode-decode will remove emojis from the title
            title = post.title.encode("ascii", "ignore").decode()
            posts.append(title)

            # If we will need dates to backtest:
            # posted_date = datetime.utcfromtimestamp(post.created_utc)


def get_subreddits_posts_pushshift() -> list[RedditPost]:
    one_year_ago = (datetime.now() - timedelta(days=365)).timestamp()
    posts = []

    for subreddit in SUBREDDITS:
        url = f"https://api.pushshift.io/reddit/search/submission?subreddit={subreddit}&after={int(one_year_ago)}"
        response = requests.get(url)

        if response.status_code == 200:
            posts = response.json()["data"]
            for post in posts:
                post_date = datetime.fromtimestamp(post.get("created_utc", 0))
                reddit_post = RedditPost(
                    post.get("title", ""),
                    post.get("ups", 0),
                    post.get("downs", 0),
                    post.get("num_comments", 0),
                    post_date,
                )
                posts.append(reddit_post)
        else:
            print(
                f"Error: received status code {response.status_code} when trying to get posts from {subreddit}"
            )

    return posts


def find_stock_interest(stock_list, timeframe="now 1-d") -> pd.DataFrame:
    pytrends = TrendReq(hl="en-US", tz=360)

    # Store the results for each stock
    results = []

    # API can only handle 5 keywords at a time, so split stocks into chunks of 5 stocks
    chunk_size = 5
    stock_chunks = [
        stock_list[i : i + chunk_size] for i in range(0, len(stock_list), chunk_size)
    ]

    # Iterate over stock chunks and retrieve the interest over time
    # Calculate interest score every 8 minutes for 24 hours, and make an average for each stock
    for chunk in tqdm(
        stock_chunks, desc=f"Getting trend scores for the stocks from Google Trends.."
    ):
        pytrends.build_payload(kw_list=chunk, timeframe=timeframe)
        interest_over_time_df = pytrends.interest_over_time()
        results.append(interest_over_time_df)
        time.sleep(1)

    # Concatenate the results and compute the average of interest across all stocks
    combined_results = pd.concat(results, axis=1)

    return combined_results.mean()


def get_ticker_dict(stocks: list[str]) -> dict[str, str]:
    """
    Get the ticker for each stock in the list
    Return a dictionary with the stock name as the key and the ticker as the value
    """
    dict = pd.read_csv("S&P_companies.csv")
    dict = dict[["Company Name", "Ticker"]]
    dict = dict.set_index("Company Name")
    dict = dict.loc[stocks]
    dict = dict.dropna()
    dict = dict.to_dict()["Ticker"]
    return dict


def get_most_talked_stocks() -> list[str]:
    """
    Get the top 20 most talked about stocks on Google
    Use Google Trends API to get the interest over the last day for each stock in S&P 500
    Return the top 20 stocks
    """
    company_names = pd.read_csv("CompanyNames.csv")
    company_names = company_names["Company Name"].to_list()

    stock_interests = find_stock_interest(company_names, timeframe="now 1-d")
    stock_interests = stock_interests.sort_values(ascending=False)

    print(
        "\n\n\nMost talked about stocks:\n",
        stock_interests.head(NUM_OF_STOCKS),
        "\n\n\n",
    )
    # save stocks to csv
    stock_interests.to_csv("most_stocks.csv")
    most_talked_about_stocks = stock_interests.head(NUM_OF_STOCKS).index.to_list()
    return most_talked_about_stocks


def calc_average_sentiment(posts: list[RedditPost]) -> float:
    # Calculate the average sentiment for each stock
    total_grade = 0
    total_posts = len(posts)
    total_weights = 0

    # Each post has a weight according to it's number of ups, downs and comments.
    for post in posts:
        post_weight = 1 + (2 * (post.ups - post.downs) + post.num_comments)
        total_grade += post_weight * post.sentiment
        total_weights += post_weight

    if total_posts == 0:
        return 5
    average_grade = total_grade / total_weights

    return average_grade


def main():
    stocks = get_most_talked_stocks()

    # We can use either PRAW api or Pushshift api to get the posts
    posts = get_subreddits_posts_praw()
    # posts = get_subreddits_posts_pushshift()

    company_name_to_ticker_dict = get_ticker_dict(stocks)
    # keep only posts that contain the stock name

    # Create a dictionary to store posts for each stock
    stock_posts = {stock: [] for stock in stocks}

    # Keep only posts that contain the stock name
    for post in posts:
        for stock in stocks:
            # Check if stock name or stock ticker is in the post title
            if (
                stock in post.title.upper()
                or company_name_to_ticker_dict[stock] in post.title.upper()
            ):
                stock_posts[stock].append(post)
                break

    # Print the posts for each stock
    print("Number of posts: ", len(posts))
    print("\n\n\n -------------------------------------------\n\n\n")
    for stock, stock_post_list in stock_posts.items():
        print(f"\n\n\nPosts for {stock}:")
        for post in stock_post_list:
            print(
                post.title,
                "ups:",
                post.ups,
                "downs:",
                post.downs,
                "num comments:",
                post.num_comments,
                sep=" ",
            )
            chatgpt_sentiment = analyze_sentiment(post.title)
            print(f"Sentiment: {chatgpt_sentiment}\n\n")
            post.sentiment = chatgpt_sentiment

    print("\n\n\n -------------------------------------------\n\n\n")
    stock_scores = {}
    for stock, stock_post_list in tqdm(
        stock_posts.items(), desc=f"Calculating average sentiment for each stock.."
    ):
        average_sentiment = calc_average_sentiment(stock_post_list)
        # Sleeping since OpenAI has a limit of requests per minute
        time.sleep(0.5)
        stock_scores[stock] = average_sentiment
    scores = pd.DataFrame(stock_scores.items(), columns=["Stock", "Score"]).sort_values(
        by="Score", ascending=False
    )

    print("Average sentiment for each stock:")
    # print and remove index
    print(scores.to_string(index=False))
    # save posts into a csv file
    df = pd.DataFrame(posts)
    df.to_csv("reddit_posts.csv", index=False)


if __name__ == "__main__":
    main()
