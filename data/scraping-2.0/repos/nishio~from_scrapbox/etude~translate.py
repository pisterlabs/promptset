SYSTEM_PROMPT = """\
As Bashi Translator, your role involves translating Japanese texts into English, \
with a focus on AI and philosophy, particularly from NISHIO Hirokazu. \
Your translations cater to readers with B1 level English proficiency, \
ensuring complex concepts are communicated clearly. \
When keywords appear in square brackets in the input, replicate this format in output, \
adding them on a new line if integration is challenging. Additionally, honor the formatting of spaces and tabs at the beginning of lines, as they indicate indentation in bullet lists. This formatting, not markdown, is crucial for preserving the original structure. All translations will be provided in a code region format for easy copy-pasting, ensuring no markdown is used. Clarify any uncertainties in the Japanese text due to the specialized nature of the content."""

from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=(SYSTEM_PROMPT)),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)

llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo", verbose=True)
# print(llm(chat_template.format_messages(text="こんにちは[世界]")))


def get_api_url(url):
    """
    >>> get_api_url("https://scrapbox.io/nishio/example")
    'https://scrapbox.io/api/pages/nishio/example'
    >>> get_api_url("https://scrapbox.io/api/pages/nishio/example")
    'https://scrapbox.io/api/pages/nishio/example'
    """
    import re

    if "api/pages/" in url:
        return url
    api_url = re.sub(
        r"(https://scrapbox\.io)/([^/]+)/([^/]+)", r"\1/api/pages/\2/\3", url
    )
    return api_url


def read_page_from_scrapbox(url):
    """
    url example: https://scrapbox.io/nishio/%F0%9F%A4%962023-08-13_07:08
    """
    import requests

    # if IS_PRIVATE_PROJECT:
    #     from read_private_project import read_private_pages

    #     page = read_private_pages(url)
    # else:
    #     api_url = get_api_url(url)
    #     page = requests.get(api_url).json()

    api_url = get_api_url(url)
    page = requests.get(api_url).json()

    return page


# page = read_page_from_scrapbox(
#     "https://scrapbox.io/nishio/100%25%E5%89%8D%E3%81%AB%E9%80%B2%E3%82%80%E3%82%B3%E3%83%88%E3%82%92%E3%82%84%E3%82%8B"
# )

# text = "\n".join([line["text"] for line in page["lines"]])

# r = llm(chat_template.format_messages(text=text))

# print(r.content)


def translate_url(url):
    page = read_page_from_scrapbox(url)
    text = "\n".join([line["text"] for line in page["lines"]])
    r = llm(chat_template.format_messages(text=text))
    return r.content
