import logging
import re
import time

import bs4
import requests
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from agent import settings

HASH_TAG_RE = re.compile("#(\w+)")


def get_data(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    try:
        response = requests.get(url, headers=headers)
        return response
    except Exception as err:
        print(err)
        return None


def get_content(data):
    soup = bs4.BeautifulSoup(data.text, "lxml")
    content = soup.select_one("#img-content")
    return content.text


def get_main_body(content):
    return content.split("\n\n")[-2]


def get_llm_chain(system_prompt, human_prompt):
    llm = ChatOpenAI(temperature=1, model="gpt-3.5-turbo-16k-0613")
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(human_prompt),
    ])

    return LLMChain(llm=llm, prompt=prompt, verbose=True)


def get_rewrite_llm_chian():
    system_prompt = "You are good at polishing article."
    human_prompt = (
        "You need to rewrite and polish below article using Chinese "
        "to make it more flowery but keep concise. "
        "The similarity between the original and the rewritten article "
        "must below 5%."
        "article: "
        "\n {article} \n"
    )
    return get_llm_chain(system_prompt, human_prompt)


def get_title_llm_chian():
    system_prompt = ("You are good at naming article. ")
    human_prompt = (
        "You need to learn the titles of articles: "
        "求求你别再舔了"
        "你要学会一松一紧"
        "我把自己弄湿了"
        "那个，你有点太快"
        "and generate 5 titles for the following article: "
        "```\n{article}\n```"
        "similar to the given exmaple and reply using Chinese."
    )
    return get_llm_chain(system_prompt, human_prompt)


def get_title(data):
    soup = bs4.BeautifulSoup(data.text, "lxml")
    title = soup.select_one("#activity-name")
    return title.text.strip()


def rewrite(url):
    data = get_data(url)
    title = get_title(data)
    logging.info(title)
    content = get_content(data)
    logging.info(content)

    with open(
        settings.WECHAT_POST_PATH / f"{time.time()}.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(title)
        f.write("\n")
        f.write(content)

    main_body = get_main_body(content)
    logging.info(main_body)
    if len(main_body) > 4000:
        return ""

    chain = get_rewrite_llm_chian()
    output = chain.run(article=main_body)
    print(output)
    return output


def rename(contents):
    if not contents:
        return ""

    chain = get_title_llm_chian()
    output = chain.run(article=contents)
    print(output)
    return output


def chain_process(url: str):
    logging.info("Start process")
    content = rewrite(url)
    name = rename(content)
    with open(settings.WECHAT_POST_PATH / f"{time.time()}.txt", "w", encoding="utf-8") as f:
        f.write(name)
        f.write("\n")
        f.write(content)
    return content
