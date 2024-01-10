#!/usr/bin/env python3
# coding=utf-8
# date 2023-04-19 11:35:53
# author calllivecn <c-all@qq.com>


import openai
import googlesearch


def google_search(query):
    results = []
    for url in googlesearch.search(query, num_results=5):
        results.append(url)
    return results

query = "How to connect OpenAI ChatGPT and Google Search"
chat_response = chat_gpt(query)
search_results = google_search(chat_response)

print("ChatGPT Response: ", chat_response)
print("Google Search Results: ", search_results)
