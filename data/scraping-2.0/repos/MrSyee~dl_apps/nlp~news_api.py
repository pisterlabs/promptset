"""Retriving News Chatbot with GPT & News API

Prompt example:
- What is the latest issues?
- What are the popular issues for September 2023?
- What is Tesla up these days?
"""

import json
import os
from typing import Any, Callable, Dict, List, Tuple

import gradio as gr
import openai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai.openai_object import OpenAIObject

load_dotenv()

TITLE_TO_URL = {}


class GPTClient:
    def __init__(self):
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.model = "gpt-3.5-turbo-0613"

    def summarize(self, texts: str) -> str:
        prompt = f"""
            Summarize the sentences '---' below.
            ---
            {texts}
            """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        # ChatGPT API 호출하기
        response = openai.ChatCompletion.create(model=self.model, messages=messages)

        input_tokens = response["usage"]["prompt_tokens"]
        output_tokens = response["usage"]["completion_tokens"]
        return response["choices"][0]["message"]["content"]

    def translate(self, texts: str) -> str:
        prompt = f"""
            Translate the sentences '---' below to Korean.
            ---
            {texts}
            """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        # ChatGPT API 호출하기
        response = openai.ChatCompletion.create(model=self.model, messages=messages)

        input_tokens = response["usage"]["prompt_tokens"]
        output_tokens = response["usage"]["completion_tokens"]
        return response["choices"][0]["message"]["content"]

    def get_args_for_function_call(
        self, messages: List[Dict[str, str]], functions: List[Dict[str, Any]]
    ) -> OpenAIObject:
        """
        If there is information for function in messages, get argument from messages.
        Otherwise get simple GPT response.
        """
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            functions=functions,
        )
        return response["choices"][0]["message"]

    def request_with_function_call(
        self,
        messages: List[Dict[str, str]],
        function: Callable,
        function_call_resp: OpenAIObject,
        prompt: str = "",
    ) -> str:
        function_name = function_call_resp["function_call"]["name"]

        if prompt:
            messages.append({"role": "system", "content": prompt})

        # Run external function
        kwargs = json.loads(function_call_resp["function_call"]["arguments"])
        function_result = function(**kwargs)

        # Append message
        messages.append(function_call_resp)
        messages.append(
            {"role": "function", "name": function_name, "content": function_result}
        )

        # GPT inference include function result
        res = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
        )
        return res["choices"][0]["message"]["content"].strip()


class NewsApiClient:
    def __init__(self):
        self.news_api_key = os.environ["NEWS_API_KEY"]
        self.max_num_articles = 5

    def get_articles(
        self,
        query: str = None,
        from_date: str = None,
        to_date: str = None,
        sort_by: str = None,
    ) -> str:
        """Retrieve articles from newsapi.org (API key required)"""
        base_url = "https://newsapi.org/v2/everything"
        headers = {"x-api-key": self.news_api_key}
        params = {
            "sortBy": "publishedAt",
            "sources": "cnn",
            "language": "en",
        }

        if query is not None:
            params["q"] = query
        if from_date is not None:
            params["from"] = from_date
        if to_date is not None:
            params["to"] = to_date
        if sort_by is not None:
            params["sortBy"] = sort_by

        # Fetch from newsapi.org
        # reference: https://newsapi.org/docs/endpoints/top-headlines
        response = requests.get(base_url, params=params, headers=headers)
        data = response.json()

        if data["status"] == "ok":
            print(
                f"Processing {data['totalResults']} articles from newsapi.org. "
                + f"Max number is {self.max_num_articles}."
            )
            return json.dumps(
                data["articles"][: min(self.max_num_articles, len(data["articles"]))]
            )
        else:
            print("Request failed with message:", data["message"])
            return "No articles found"


news_api_client = NewsApiClient()
gpt_client = GPTClient()


def scrap_cnn_article(title: str) -> Tuple[str, str]:
    url = TITLE_TO_URL[title]
    rep = requests.get(url)

    soup = BeautifulSoup(rep.content, "html.parser")

    # Get main contents
    article = ""
    for paragraph in soup.find_all(["p", "h2"], {"class": ["paragraph", "subheader"]}):
        article += paragraph.text.strip()

    # Summarize and translate to Korean
    summarized_article = gpt_client.summarize(article)
    translated_article = gpt_client.translate(summarized_article)

    return summarized_article, translated_article


signature_get_articles = {
    "name": "get_articles",
    "description": "Get news articles",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Freeform keywords or a phrase to search for.",
            },
            "from_date": {
                "type": "string",
                "description": "A date and optional time for the oldest article allowed. This should be in ISO 8601 format",
            },
            "to_date": {
                "type": "string",
                "description": "A date and optional time for the newest article allowed. This should be in ISO 8601 format",
            },
            "sort_by": {
                "type": "string",
                "description": "The order to sort the articles in",
                "enum": ["relevancy", "popularity", "publishedAt"],
            },
        },
        "required": [],
    },
}

signature_get_title_and_url = {
    "name": "get_title_and_url",
    "description": "Get title of article and url.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "array",
                "description": "title array of articles",
                "items": {"type": "string", "description": "title of article"},
            },
            "url": {
                "type": "array",
                "description": "url array of articles",
                "items": {"type": "string", "description": "url of article"},
            },
        },
        "required": ["title", "url"],
    },
}


def respond(prompt: str, chat_history: List[str]) -> Tuple[str, List[str]]:
    global TITLE_TO_URL

    # Get args from prompt
    messages = [{"role": "user", "content": prompt}]
    args_resp = gpt_client.get_args_for_function_call(messages, [signature_get_articles])

    # call functions requested by the model
    answer = args_resp["content"]
    title_list = []
    if args_resp.get("function_call"):
        # GPT inference again with calling external function
        get_articles_prompt = """
            You are an assistant that provides news and headlines to user requests.
            Always try to get the articles using the available function calls.
            Write the arguments to your function at the top of your answer.
            Please output something like this:
            Number. [Title](Article Link)\n
                - Description: description\n
                - Publish Date: publish date\n
        """
        answer = gpt_client.request_with_function_call(
            messages=messages,
            function=news_api_client.get_articles,
            function_call_resp=args_resp,
            prompt=get_articles_prompt,
        )

        # Get titles and urls for dropdown from response message
        messages = [{"role": "user", "content": answer}]
        args_resp = gpt_client.get_args_for_function_call(
            messages, [signature_get_title_and_url]
        )
        args = json.loads(args_resp["function_call"]["arguments"])
        title_list, url_list = args.get("title"), args.get("url")
        TITLE_TO_URL = {title: url for title, url in zip(title_list, url_list)}

    chat_history.append((prompt, answer))

    # Update dropdown
    drop_down = None
    if title_list:
        drop_down = gr.update(choices=title_list, interactive=True)

    return "", chat_history, drop_down


with gr.Blocks() as demo:
    gr.Markdown("# 뉴스 기사 탐색 챗봇")
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                ## Chat
                얻고 싶은 정보에 대해 질문해보세요.
                """
            )
            chatbot = gr.Chatbot(label="Chat History")
            prompt = gr.Textbox(label="Input prompt")
            clear = gr.ClearButton([prompt, chatbot])

        with gr.Column():
            gr.Markdown(
                """
                ## Select News article
                원하는 기사를 선택하세요.
                """
            )
            article_list = gr.Dropdown(label="Article List", choices=None)
            abstract_box = gr.Textbox(
                label="Summarized article", lines=10, interactive=False
            )
            translate_box = gr.Textbox(
                label="Translated article", lines=10, interactive=False
            )
            scrap_btn = gr.Button("Get article!")

    prompt.submit(respond, [prompt, chatbot], [prompt, chatbot, article_list])
    scrap_btn.click(
        scrap_cnn_article, inputs=[article_list], outputs=[abstract_box, translate_box]
    )


if __name__ == "__main__":
    demo.launch()
