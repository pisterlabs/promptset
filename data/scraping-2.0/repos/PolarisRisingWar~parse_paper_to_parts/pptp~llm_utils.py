import openai

from keys import *

openai.api_key=OPENAI_API_KEY
openai.proxy=OPENAI_PROXY

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

@retry(wait=wait_random_exponential(min=20, max=600),stop=stop_after_attempt(6))
def first_page_title(first_page_text):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "请直接输出如下抽取文本中的中文论文标题，不要带其他任何内容："+first_page_text}
        ]
    )

    return completion.choices[0].message['content']

@retry(wait=wait_random_exponential(min=20, max=600),stop=stop_after_attempt(6))
def from_catalog_extract_titles(catalog_text):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user",
             "content": "请直接输出如下目录中的标题内容，要求从第一章绪论开始，格式类似于['第一章 绪论','1.1 研究背景及意义','1.2 实验数据集']这样的Python列表文本（可以直接用Python的`eval()`函数转换为列表对象的字符串），要求列表中的元素是完全出自原文的，包含序号（可能为第一章或者第1章等写法）和标题本体（如绪论或研究背景等）\n"\
                        +catalog_text}
        ]
    )

    return completion.choices[0].message['content']

@retry(wait=wait_random_exponential(min=20, max=600),stop=stop_after_attempt(6))
def from_image_text_extract_description_and_id(input_image):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user",
             "content": "请基于以下文本，如果这段文本描述的是一张单图，则直接打印描述该图片的文字内容和对应序号（举例：文字内容是“图 1-2 Doc2vec 添加段落特征预测下一个词”，对应序号是“图 1-2”，最终的输出结果应为['图 1-2','图 1-2 Doc2vec 添加段落特征预测下一个词']，要求输出可以通过Python的eval()函数直接转化为列表对象的字符串，不要包含换行符），如果这段文本描述的是一张图片的多张子图，则直接打印大图的文字内容和对应序号：\n"\
                        +input_image}
        ],
        temperature=1
    )

    return completion.choices[0].message['content']