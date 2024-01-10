import json

import openai
import pymysql

from chat.wechat_server import WechatServer
from chat.lib.itchat.utils import logger



def cal_time(func):
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        logger.info("函数名：%s\t执行时间：%s", func.__name__, str(end - start))
        return res

    return wrapper


def connect_mysql():
    try:
        # 读取配置文件
        with open('config.json', encoding='utf-8') as f:
            config_data = json.load(f)
        # 创建连接对象
        mysql_host = config_data['mysql_host']
        mysql_username = config_data['mysql_username']
        mysql_password = config_data['mysql_password']
        mysql_databases = config_data['mysql_databases']
        connection = pymysql.connect(
            host=mysql_host,  # 数据库主机地址
            user=mysql_username,  # 数据库用户名
            password=mysql_password,  # 数据库密码
            database=mysql_databases,  # 数据库名称
            charset='utf8'
        )

        # 创建游标对象
        cursor = connection.cursor()
        return cursor
    except Exception as e:
        logger.info(f'数据库连接失败{e}')


# 获取指定用户名用户单词数据
@cal_time
def get_user_word(user_name):
    cursor = connect_mysql()
    sql = "select current_word from users where username = '%s'" % user_name
    cursor.execute(query=sql)
    return cursor.fetchone()[0]


# 根据get_user_word返回的单词，获取单词详细信息
@cal_time
def get_word_detail(word):
    cursor = connect_mysql()
    sql = "select * from words where word = '%s'" % word
    cursor.execute(query=sql)
    return cursor.fetchone()


# 根据get_user_word返回的单词，获取后十个单词
@cal_time
def get_word_detail_ten(word):
    cursor = connect_mysql()
    # 获取符合条件的后面十条数据
    ten_words_query = "SELECT * FROM words WHERE word >= %s LIMIT 10"
    cursor.execute(ten_words_query, (word,))
    return cursor.fetchall()


# 对get_word_detail_ten返回10个单词随机返回一个单词
@cal_time
def get_random_word(ten_words):
    import random
    random_word = random.choice(ten_words)
    return random_word


# 把get_random_word返回的单词发送给chatgpt，让chatgpt担任英语老师，利用这个单词对我进行提问
context = ""
# 读取配置文件
with open('config.json', encoding='utf-8') as f:
    config_data = json.load(f)

model = config_data['model']

openai.api_key = config_data['open_ai_api_key']
openai.api_base = config_data['proxy']
history = []


@cal_time
def check_word(word):

    prompt = f"你现在是一名资深的英语老师人工智能。你只需要回答一切与单词相关的问题，其余一概不回答。" \
             f"我给你利用程序给你提供一条单词，你提问的时候需要加上输出这个单词{str(word)}的单词部分，而不是只提问用户知道{word[1]}的翻译吗" \
             f"，询问用户是否知道{word[1]}的英文翻译，而不能透露这个单词的翻译结果" \
             f"然后用户会回答你这个单词的中文翻译，你需要严谨判断用户回答的翻译是否与英文单词原意相同，必须严谨。" \
             f"然后回答用户正确或错误。如果用户答对了，你需要给予用户鼓励。如果用户答错了，则你需要对用户进行指导举例解释说明等。" \
             f"我提供给你的单词是：{word}。"
    logger.info(word)
    try:
        chat_completion = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[{"role": "user", "content": prompt}],
            timeout=15
        )
        reuest = chat_completion.choices[0].message.content
        logger.info(reuest)
        reuest += f"{word[1]}"
        history.append(prompt)
        history.append(reuest)
        return reuest
    except openai.error.OpenAIError as e:
        return "OpenAi调用超时请重试"

# 获取用户输入的单词
@cal_time
def get_user_input(ran_word, results):
    prompt = f"你现在是一名资深的英语老师人工智能。你只需要回答一切与单词相关的问题，其余一概不回答。" \
             f"你需要检测用户给你的翻译是否正确，" \
             f"你需要严谨判断用户回答的翻译是否与英文单词原意相同，必须严谨。" \
             f"然后回答用户正确或错误。如果用户答对了，你需要给予用户鼓励。如果用户答错了或说不知道，则你需要对用户进行指导,用但单词解释举例等。" \
             f"我提供给你的英文单词是：{str(ran_word[1])}，用户的回答是{results}。"

    logger.info(ran_word[1])
    logger.info(results)
    try:
        chat_completion = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[{"role": "user", "content": prompt}],
            timeout=15
        )

        reuest = chat_completion.choices[0].message.content

        logger.info(reuest)
        history.append(prompt)
        history.append(reuest)

        return reuest
    except openai.error.OpenAIError as e:
        return "OpenAi超时，请重试"

def get_word(nickname):
    word = get_user_word(nickname)
    print(word)
    print(get_word_detail(word))
    ten_words = get_word_detail_ten(word)
    print(ten_words)
    random_word = get_random_word(ten_words)
    print(type(random_word))
    return random_word


@cal_time
def main(flag,random_word ,reuest = None):

        if flag:
            ck_r = check_word(random_word)
            print(ck_r)
            return ck_r

        else:
            su = (get_user_input(random_word,reuest))
            return su



