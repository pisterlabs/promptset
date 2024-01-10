import openai
import requests
import init
from loguru import logger
from difflib import SequenceMatcher


@logger.catch
def ping_test(part):
    url = "https://xiaoapi.cn/API/sping.php?url=" + part
    response = requests.get(url, proxies={
        'http': None,
        'https': None}).text
    return response


@logger.catch
def chatgpt_reply(part):
    if not init.redis_handle.check_key(part):
        chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                       messages=[{"role": "user", "content": part}])
        response_txt = chat_completion['choices'][0]['message']['content']
        init.redis_handle.set(part, response_txt)
        logger.debug('To cache text')
        return response_txt
    else:
        logger.debug('Return Text from cache')
        return init.redis_handle.get(part).decode('utf-8')


@logger.catch
def hot_news():
    url = "https://xiaoapi.cn/API/zs_xw.php?num=20"
    news_txt = requests.get(url, proxies={
        'http': None,
        'https': None}).json()['msg']
    return news_txt


@logger.catch
def random_txt():
    url = "https://v1.hitokoto.cn/"
    txt = requests.get(url).json()['hitokoto']
    return txt


@logger.catch
def is_similar(prompt1, prompt2):
    # 使用SequenceMatcher计算文本相似度
    similarity = SequenceMatcher(None, prompt1, prompt2).ratio()
    return similarity > 0.8  # 相似度阈值可以根据需求调整
