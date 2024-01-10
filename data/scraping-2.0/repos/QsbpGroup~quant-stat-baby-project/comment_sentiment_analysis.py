import os
import re
import time
import openai
import random
import argparse
import requests
import pandas as pd
from tqdm import tqdm


openai.api_key = os.getenv("OPENAI_API_KEY")


def crawler(stock_code='CSI000941', n_pages=100, save=False):
    """
    Crawler for xueqiu.com

    Input:
    ------
    url: url for xueqiu.com
    save: save data or not

    Output:
    -------
    return: comments(list)
    """

    url_type = {
        'SH': 13,
        'SZ': 11,
        'CS': 26
    }
    url_prefix_comment = 'https://xueqiu.com/query/v1/symbol/search/status.json?count=10&comment=0&symbol='

    url_response = []
    n_pages = int(n_pages)
    for page in tqdm(range(1, n_pages+1), desc='Crawling', leave=False):
        url = url_prefix_comment+stock_code+'&hl=0&source=all&sort=alpha&page=' + \
            str(page)+'&q=&type='+str(url_type.get(stock_code[:2], 26))
        response_json = _get_response(url)
        if response_json is None:
            continue
        url_response.extend(response_json['list'])
    data_list = _get_comment(url_response)
    data_df = pd.DataFrame(data_list, columns=['text', 'comment_time', 'title', 'like_count', 'reply_count', 'favorite_count',
                                                   'view_count', 'retweet_count', 'is_hot', 'is_answer', 'is_bonus', 'is_reward', 
                                                   'reward_count', 'user_id', 'screen_name', 'followers_count', 'tag'])

    if save:
        data_df.to_csv("./comment_data.csv", encoding="utf_8_sig",
                       index=False, header=True)
    return data_df


def _get_response(url, headers=''):
    if headers == '':
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
            "Cookie": "cookiesu=591694835194054; device_id=02e9e5105707187692a3ebf043d62941; remember=1; xq_is_login=1; u=8176314854; s=ab12mnrdfx; bid=f24325f9c5bb92500d7f9d541ef6ef8f_lmra6p3v; __utmz=1.1695188801.2.2.utmcsr=github.com|utmccn=(referral)|utmcmd=referral|utmcct=/SJTUMisaka/xueqiu; __utma=1.486665064.1695186225.1695587344.1695604057.4; xq_a_token=76b609375630ee3af674d6ff1312edcc54cda518; xqat=76b609375630ee3af674d6ff1312edcc54cda518; xq_id_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ1aWQiOjgxNzYzMTQ4NTQsImlzcyI6InVjIiwiZXhwIjoxNjk5MTk3MDMxLCJjdG0iOjE2OTY2MDUwMzExMzcsImNpZCI6ImQ5ZDBuNEFadXAifQ.f7xKDW5MpDFMH2Opwn90zwIVVTsSZDcM8BT12a_ID-SfjvDJabSF-i7iejn5UH2TGmfdHT3uJjG8tEwphtUZGhqT4wB1cQI6jOtAToMRnTPjEIlM4_FYrFCL9KyxltsL2HE75AzoZiNYrx9L4JYWaTHwVb8EyOlxZJCb7azWIajJvEgPbKOJODA25J9iu5qmankMpG0RcGHeVvajJbZyt-yU1rTJI8LEeo_RsxgBIxJg9K5HiiMkWs3VNkyXhqqZ5mHxRMaT7Fl5XAT1kRorW799DJBpwFZhY0fNNtNB7B0D0EUL5fBENGzKVGrUGu9QTGkVLZNGpFvIB4ACnXJ8Gg; xq_r_token=034012b5249fa1ae316050a7251e6d9a403ea76b; Hm_lvt_1db88642e346389874251b5a1eded6e3=1695402118,1695435276,1695587332,1696605033; snbim_minify=true; Hm_lpvt_1db88642e346389874251b5a1eded6e3=1696631105; acw_tc=2760779616966318847376306e9556dc4f4ed8169fb3c338239efbc1e25e52"
        }
    max_retries = 3
    for i in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            response_json = response.json()
            return response_json
        except requests.RequestException as e:
            time.sleep(3)
    return None

def _fetch_user_followers_count(user_id):
    """
    input: user_id
    output: followers_count
    """
# 为了减轻连接压力，添加适当的延迟
    url_prefix_user = 'https://xueqiu.com/statuses/original/show.json?user_id='
    url_user = url_prefix_user + str(user_id)
    
    time.sleep(0.3)
    max_retries = 3
    
    for i in range(max_retries):
        try:
            response = _get_response(url_user)
            followers_count = int(response['user']['followers_count'])
            return followers_count
        except requests.RequestException as e:
            time.sleep(3)
    return -1


def _get_comment(data):
    """
    Get comments from xueqiu.com, includingtext, comment_time, title, like_count, reply_count, favorite_count, view_count, retweet_count, is_hot, is_answer, is_bonus, is_reward, reward_count, screen_name

    Input:
    ------
    data: data from xueqiu.com

    Output:
    -------
    return: comments
    """
    data_list = []
    pinglun_len = len(data)
    print('Number of comments:', pinglun_len)

    for i in tqdm(range(pinglun_len), desc='Extracting comments', leave=False):
        temp_data = data[i]
        
        user_id = temp_data['user_id']
        followers_count = _fetch_user_followers_count(user_id)
        tag = ['路人', '大牛'][followers_count > 10000]
        
        des = '>' + temp_data['description'] + '<'
        pre = re.compile('>(.*?)<')
        text = ''.join(pre.findall(des))
        # convert timestamp into real time
        timeArray = time.localtime(temp_data['created_at'] / 1000 + 11*3600)
        comment_time = time.strftime("%Y-%m-%d %H:%M", timeArray)
        title = temp_data['title']
        like_count = temp_data['like_count']
        reply_count = temp_data['reply_count']
        favorite_count = temp_data['fav_count']
        view_count = temp_data['view_count']
        retweet_count = temp_data['retweet_count']
        is_hot = temp_data['hot']
        is_answer = temp_data['is_answer']
        is_bonus = temp_data['is_bonus']
        is_reward = temp_data['is_reward']
        reward_count = temp_data['reward_count']
        screen_name = temp_data['user']['screen_name']
        data_list.append([text, comment_time, title, like_count, reply_count, favorite_count, view_count,
                          retweet_count, is_hot, is_answer, is_bonus, is_reward, reward_count, user_id, screen_name, followers_count, tag])
    return data_list


# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specific errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def get_sentiment(comment, model_choose):
    """
    Get sentiment from comment

    Input:
    ------
    prompt: prompt for openai

    Output:
    -------
    return: sentiment(one of Bullish, Bearish, Neutral)
    """
    system_prompt = """ 
    Role: Pretend you are an experienced stock market manager. You are good at analysing sentiment from the Chinese stock market forum.
    Background: The user will provide you with a snippet of discussion from a stock forum regarding a specific stock or sector. Your task is to evaluate the sentiment expressed by the individual.
    Output Format: reply only one of the following: Bullish, Bearish, or Neutral. 
    Note: Prioritize determining whether the sentiment is Bullish or Bearish; only use "Neutral" if the sentiment is genuinely ambiguous or unclear.
    """
    model=["gpt-3.5-turbo", "gpt-4"][model_choose == 'gpt-4']
    
    times = 0
    while True:
        times += 1
        response = chat_completions_with_backoff(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": comment},
            ]
        )
        sentiment = response['choices'][0]['message']['content']
        # count words
        if sentiment in ['Bullish', 'Bearish', 'Neutral']:
            break
        else:
            if times > 2:
                sentiment = 'Neutral'
                break
            continue
    return sentiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crawler and sentiment analysis for xueqiu.com')
    parser.add_argument('-n', '--n_pages', default=100, type=int, help='number of comments pages to crawl')
    parser.add_argument('-s', '--sentiment', default='True', type=str, help='determine whether to get sentiment')
    parser.add_argument('-g', '--gpt', default='gpt-4', type=str, help='gpt model to use')
    args = parser.parse_args()
    file_name = 'comment_data.csv'
    if os.path.exists(os.path.join(os.getcwd(), file_name)):
        comment_df = pd.read_csv(file_name)
        if len(comment_df) < 900:
            comment_df = crawler(n_pages=args.n_pages, save=True)
    else:
        comment_df = crawler(n_pages=args.n_pages, save=True)
    if args.sentiment not in ['False', 'false', 'FALSE', 'F', 'f', '0', 'no', 'No', 'NO', 'n', 'N']:
        sentiments = []
        for i in tqdm(range(len(comment_df)), desc='Getting sentiment', leave=False):
            # make sure not to exceed the rate limit of API
            if (i+1) % 20 == 0:
                time.sleep(10)
            comment = comment_df['text'][i]
            sentiment = get_sentiment(comment, model_choose=args.gpt)
            sentiments.append(sentiment)
        comment_df['sentiment'] = sentiments
        comment_df.to_csv("./comment_data_with_sentiment.csv",
                        encoding="utf_8_sig", index=False, header=True)