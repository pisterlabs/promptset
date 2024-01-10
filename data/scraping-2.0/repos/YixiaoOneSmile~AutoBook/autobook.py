import openai
import os
import json

with open('config.json', 'r') as file:
    config = json.load(file)

API_KEY = config['OPENAI']['API_KEY']
OPENAI_BASE_URL = config['OPENAI']['BASE_URL']

class Autobook:

    BASE_PROMPT = """请你扮演一个{book_name}专家，你非常擅长写作，你能将知识详细的生动的写出来"""
    OUTLINE_PROMPT = """
        请为我生成一个关于{book_name}markdown格式图书大纲，
        只可以生成一级目录和二级目录，
        # 代表一级目录，## 代表二级目录，
        除了章标题和节标题以外，不需要具体内容
        例如：
        #书名
        ## 第一章：xxx
        ### 1.1 xxx
        ### 1.2 xxx
        ## 第一章：xxx
        ### 2.1 xxx
        ### 1.2 xxx
    """
    SECTION_PROMPT = """这是关于{book_name}书籍{chapter_name}{section_title}的内容，
                    你要请为我用markdown格式生成此节的具体内容，
                    你要让文字写的富有逻辑性，
                    你要将此节的内容写的生动有趣，让读者能够很好的理解，
                    你要尽量详细的列举出关于此节的所有知识点，并对这些知识点进行详细的解释，
                    """

    def __init__(self):
        openai.api_key = API_KEY
        os.environ["OPENAI_API_BASE"] = OPENAI_BASE_URL

    def _generate(self, messages):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content

    def generate_outline(self, book_name):
        messages = [
            {"role": "system", "content": self.BASE_PROMPT.format(book_name=book_name)},
            {"role": "user", "content": self.OUTLINE_PROMPT.format(book_name=book_name)}
        ]
        return self._generate(messages)

    def generate_section_content(self, book_name, chapter_name, section_title):
        messages = [
            {"role": "system", "content": self.BASE_PROMPT.format(book_name=book_name)},
            {"role": "user", "content": self.SECTION_PROMPT.format(book_name=book_name, chapter_name=chapter_name, section_title=section_title)}
        ]
        return self._generate(messages)