import os
import re
import ast
# News API
from newsapi import NewsApiClient
# OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
# Crawling
from news_crawling import download_news_content
# Secrets
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
newsapi_api_key = os.environ.get("NEWSAPI_API_KEY")

# Init News API
newsapi = NewsApiClient(api_key=newsapi_api_key)


async def get_top_headlines():
    # /v2/top-headlines
    top_headlines = newsapi.get_top_headlines(page_size=100)
    # Get all the news titles from the response
    list_news_titles_top_headlines = [article['title'] for article in top_headlines['articles']]
    # Convert the list of news titles to a single string
    text_news_titles_top_headlines = ''
    for index, news_title in enumerate(list_news_titles_top_headlines):
        text_news_titles_top_headlines += str(index + 1) + '. ' + news_title + '\n'
    return top_headlines, text_news_titles_top_headlines


async def get_openai_response_of_extracting_news():
    chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name='gpt-4-1106-preview')

    #
    top_headlines, text_news_titles_top_headlines = await get_top_headlines()

    news_prompt = "You are a professional news editor and your task is to filter news headlines above and return the news numbers that meet the requirements of a specified topic. The specified topics are: Global Emergency News, AI and Technology Frontier, Finance and Economy. Your return result can only contain news numbers that meet the requirements of the specified topic in list-like format (e.g. [1, 3, 5, 8]). DO NOT output the original title nor anything else."

    res = chat(
        [
            HumanMessage(content=text_news_titles_top_headlines + '\n\n' + news_prompt)
        ]
    )

    print(res.content)

    res_content = res.content

    pattern = r"\[(.+?)\]"
    lists = re.findall(pattern, res_content)
    combined_list = []

    for lst in lists:
        if lst:
            result = ast.literal_eval(lst)
            if isinstance(result, int):  # Check if the result is an integer
                combined_list.append(result)  # If it is, append to the list
            else:
                combined_list.extend(result)  # If not, extend the list

    print(combined_list)

    list_news_top_headlines = [article for article in top_headlines['articles']]
    # Extract items using list comprehension
    extracted_list = [list_news_top_headlines[i - 1] for i in combined_list]
    return extracted_list


async def summarize_news_in_chinese(news_data):
    # Download the whole page content of news_data['url']
    article_content = await download_news_content(news_data['url'])

    if article_content:
        summarization_prompt = (
                article_content + """\n\nYou are a professional news editor and your task is to summarise the articles in the news page above. The articles may be written in a language such as English, but your summary is to be done in Chinese (Simplified). Your return result should only contain the news headlines and summary content in the format below:\n<CHINESE TITLE>\n<SUMMARY>\nPlease do not output any other content."""
        )

        chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name='gpt-3.5-turbo-1106')

        # Use OpenAI to read the whole page and output the summary of this page in Chinese
        res = chat(
            [
                HumanMessage(content=summarization_prompt)
            ]
        )

        # Assuming that the response successfully contains a summary in Chinese
        summary_in_chinese = res.content

        return article_content, summary_in_chinese
    else:
        return "无法下载新闻内容", "无法提供新闻摘要"  # "Unable to download the article content, so cannot provide a summary."


async def generate_news_comment(news_page_content):
    comment_prompt = (
            news_page_content + """\n\nAs a knowledgeable, educated and professional news commentator, your task is to make a short comment on the above news, analysing the implications of the news from various perspectives and then give your final views. Please note, that you need to provide substantial comments, not empty rhetoric. You should fully utilize analytical thinking, instead of using too many exaggerated adjectives to elaborate your argument. When you list the sub-points, please use the form of bullet points to present them. Give your comment in Chinese (Simplified)."""
    )

    chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name='gpt-3.5-turbo-1106')

    # Use OpenAI to read the whole page and output the summary of this page in Chinese
    res = chat(
        [
            HumanMessage(content=comment_prompt)
        ]
    )

    # Assuming that the response successfully contains a summary in Chinese
    news_comment = res.content

    return news_comment


async def clean_brackets(original_string):
    if original_string.startswith("《") and original_string.endswith("》") or \
            original_string.startswith("<") and original_string.endswith(">"):
        return original_string[1:-1]
    else:
        return original_string
