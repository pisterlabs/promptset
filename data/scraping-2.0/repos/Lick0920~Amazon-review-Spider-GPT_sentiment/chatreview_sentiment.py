import openai
import re
import pandas as pd
import nltk

# 读取Excel文件
# file_name = '/home/changkang.li/papers/Amazon-review-crawler/AmazonReviewSpider/评论数据/1_评论数据2023-03-23 10-30-52total.xlsx'
file_name = '/home/changkang.li/papers/nlp_textblob/everthing.xlsx'
# file_name = '/home/changkang.li/papers/nlp_textblob/test.xlsx'
# file_name = '/home/changkang.li/papers/nlp_textblob/miss_heart_300_sort.xlsx'
data = pd.read_excel(file_name, skiprows=1)
reviews = data['内容'].tolist()
BOOK_REVIEW_THRESHOLD = 5000
# 设置OpenAI API密钥
openai.api_key = "your key"

import os
# 定义GPT模型ID
model_engine = "text-davinci-003"
# model_engine = "gpt-3.5-turbo-0301"

# 定义函数来分析评论
def analyze_review(review):
    # 使用OpenAI的GPT模型来分析评论
    prompt =  "filter out the complete sentences from the review below that express the reviewer's sentiment about the book, regardless of whether the sentiment is positive, negative, or neutral. These sentences may contain emotion-suggestive verbs, nouns, adjectives, adverbs, punctuation, tone, metaphor, or any other expressions. ensure that any extracted sentences are complete.With the complete sentences,  give them a score from 1 (very negative) to 5 (very positive) based on the reviewer's sentiment. You should put the score within the brackets behind each sentence. Do not separate two tasks into two parts of your answer. Negation may occur in negative sentences.write the score at the end of the sentence with[]\n"
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt + review,
        max_tokens=2048,
        n=1,
        stop=None,
        temperature=0.5,
    )
    # response = openai.Completion.create(
    #     # engine="davinci",  # changed from "gpt-3.5-turbo"
    #     prompt=prompt + review,
    #     max_tokens=1024,
    #     n=1,
    #     stop=None,
    #     temperature=0.5,
    #     model="gpt-3.5-turbo" # Match the model of the chosen endpoint.
    # )

    # 获取GPT模型的响应
    sentiment_analysis = response.choices[0].text.strip()
    print(sentiment_analysis)
    # 返回结果
    return {
        "book_sentiment": sentiment_analysis,
    }

# 分析每个评论
from tqdm import tqdm


# 初始化book_sentiments列表
book_sentiments = []

combined_review = ''
# 遍历每个评论，进行情感分析
for i, review in tqdm(enumerate(reviews)):
    # if i < 49:
    #     continue
    if i < 2037:
        continue

    combined_review += review
    # 如果组合的评论长度达到500或者已经遍历完所有评论
    if len(combined_review) >= BOOK_REVIEW_THRESHOLD or i == len(reviews) - 1:
        # 如果评论长度长于1000，则将其拆为两个评论，分别进行情感分析
        if len(combined_review) > 8000:
            # 将评论拆分成多个句子
            sentences = nltk.sent_tokenize(combined_review)
            sentence_count = len(sentences)
            mid_sentence_index = sentence_count // 2
            # 将评论分成两个部分
            review1 = ' '.join(sentences[:mid_sentence_index])
            review2 = ' '.join(sentences[mid_sentence_index:])
            # 分别进行情感分析
            result1 = analyze_review(review1)
            print(result1)
            result2 = analyze_review(review2)
            print(result2)
            # 将两个结果合并
            result = {}
            result['book_sentiment'] = result1['book_sentiment'] + result2['book_sentiment']
    
            with open('everything_2037.txt', 'a') as f:
                f.write(result['book_sentiment'] + '\n')
            print("save to everything_2037.txt")
        else:
            result = analyze_review(combined_review)
            with open('everything_2037.txt', 'a') as f:
                f.write(result['book_sentiment'] + '\n')
            print("save to everything_2037.txt")
        
        combined_review = ''

        # 将result中的book_sentiment分成多个句子，然后保存到book_sentiments.txt文本中
        # import nltk
        # nltk.download('punkt')

        # # 将result中的book_sentiment分成多个句子，然后保存到book_sentiments.txt文本中
        # book_sentiment = result['book_sentiment']
        # sentences = nltk.sent_tokenize(book_sentiment)
        # with open('firstbook_first1k_retry.txt', 'a') as f:
        #     for sentence in sentences:
        #         if sentence.strip() != '':
        #             f.write(sentence.strip() + '\n')
        # print("save to firstbook_first1k_retry.txt")
        # # 不分句子了 直接保存
        # with open('Little_fire.txt', 'a') as f:
        #     f.write(result['book_sentiment'] + '\n')
        # print("save to Little_fire.txt")

print("all end")

