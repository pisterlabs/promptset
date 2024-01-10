import datetime
import os
import re

import aiohttp
import openai
import requests
import tenacity
import tiktoken
from bs4 import BeautifulSoup
from loguru import logger
from pydantic import BaseModel

from ..paper_with_image import Paper
from ..provider import async_arxiv as arxiv
from ..utils import load_config, report_token_usage


class ArxivParams(BaseModel):
    query: str
    key_word: str
    page_num: int
    max_results: int
    days: int
    sort: str
    save_image: bool
    file_format: str
    language: str


class Reader:
    # 初始化方法，设置属性
    def __init__(
        self,
        key_word,
        query,
        root_path="./",
        sort=arxiv.SortCriterion.SubmittedDate,
        user_name="defualt",
        args=None,
    ):
        self.user_name = user_name  # 读者姓名
        self.key_word = key_word  # 读者感兴趣的关键词
        self.query = query  # 读者输入的搜索查询
        self.sort = sort  # 读者选择的排序方式
        self.args = args

        if args is None:
            raise ValueError("args cannot be None")

        if args.language == "en":
            self.language = "English"
        elif args.language == "zh":
            self.language = "Chinese"
        else:
            self.language = "English"

        self.root_path = root_path
        self.config, self.chat_api_list = load_config()

        self.cur_api = 0
        self.file_format = args.file_format

        if args.save_image:
            self.gitee_key = self.config["Gitee"]["api"]
        else:
            self.gitee_key = ""

        self.max_token_num = 4096
        self.encoding = tiktoken.get_encoding("gpt2")

    # 定义一个函数，根据关键词和页码生成arxiv搜索链接
    def get_url(self, keyword, page):
        base_url = "https://arxiv.org/search/?"
        params = {
            "query": keyword,
            "searchtype": "all",  # 搜索所有字段
            "abstracts": "show",  # 显示摘要
            "order": "-announced_date_first",  # 按日期降序排序
            "size": 50,  # 每页显示50条结果
        }
        if page > 0:
            params["start"] = page * 50  # 设置起始位置

        return base_url + requests.compat.urlencode(params)

    # 定义一个函数，根据链接获取网页内容，并解析出论文标题
    async def get_titles(self, session: aiohttp.ClientSession, url, days=1):
        titles = []
        # 创建一个空列表来存储论文链接
        links = []
        dates = []

        async with session.get(url) as response:
            text = await response.text()
            soup = BeautifulSoup(text, "html.parser")
            articles = soup.find_all("li", class_="arxiv-result")  # 找到所有包含论文信息的li标签
            today = datetime.date.today()
            last_days = datetime.timedelta(days=days)
            for article in articles:
                title = article.find("p", class_="title").text  # 找到每篇论文的标题，并去掉多余的空格和换行符
                link = article.find("span").find_all("a")[0].get("href")
                date_text = article.find("p", class_="is-size-7").text
                date_text = (
                    date_text.split("\n")[0].split("Submitted ")[-1].split("; ")[0]
                )
                date_text = datetime.datetime.strptime(date_text, "%d %B, %Y").date()
                if today - date_text <= last_days:
                    titles.append(title.strip())
                    links.append(link)
                    dates.append(date_text)
            return titles, links, dates

    # 定义一个函数，根据关键词获取所有可用的论文标题，并打印出来
    async def get_all_titles_from_web(self, keyword, page_num=1, days=1):
        title_list, link_list, date_list = [], [], []
        for page in range(page_num):
            url = self.get_url(keyword, page)  # 根据关键词和页码生成链接
            titles, links, dates = await self.get_titles(url, days)  # 根据链接获取论文标题
            if not titles:  # 如果没有获取到任何标题，说明已经到达最后一页，退出循环
                break
            for title_index, title in enumerate(titles):  # 遍历每个标题，并打印出来
                logger.info(
                    f"{page}, {title_index}, {title}, {links[title_index]}, {dates[title_index]}"
                )
            title_list.extend(titles)
            link_list.extend(links)
            date_list.extend(dates)
        logger.info("-" * 40)
        return title_list, link_list, date_list

    def get_arxiv(self, max_results=30):
        search = arxiv.Search(
            query=self.query,
            max_results=max_results,
            sort_by=self.sort,
            sort_order=arxiv.SortOrder.Descending,
        )
        return search

    async def get_arxiv_web(self, args, page_num=1, days=2):
        titles, links, dates = await self.get_all_titles_from_web(
            args.query, page_num=page_num, days=days
        )
        paper_list = []
        for title_index, title in enumerate(titles):
            if title_index + 1 > args.max_results:
                break
            logger.info(
                f"{title_index}, {title}, {links[title_index]}, {dates[title_index]}"
            )
            url = links[title_index] + ".pdf"  # the link of the pdf document

            filename = self.try_download_pdf(url, title)

            paper = Paper(
                path=filename,
                url=links[title_index],
                title=title,
            )
            paper_list.append(paper)
        return paper_list

    def validateTitle(self, title):
        # 将论文的乱七八糟的路径格式修正
        rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
        new_title = re.sub(rstr, "_", title)  # 替换为下划线
        return new_title

    def download_pdf(self, url, title):
        response = requests.get(url)  # send a GET request to the url
        date_str = str(datetime.datetime.now())[:13].replace(" ", "-")
        path = (
            self.root_path
            + "pdf_files/"
            + self.validateTitle(self.query)
            + "-"
            + date_str
        )

        try:
            os.makedirs(path)
        except Exception:
            pass
        filename = os.path.join(path, self.validateTitle(title)[:80] + ".pdf")
        with open(filename, "wb") as f:  # open a file with write and binary mode
            f.write(response.content)  # write the content of the response to the file
        return filename

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    def try_download_pdf(self, url, title):
        return self.download_pdf(url, title)

    def summary_with_chat(self, paper_list):
        htmls = []
        for paper_index, paper in enumerate(paper_list):
            # 第一步先用title，abs，和introduction进行总结。
            text = ""
            text += "Title:" + paper.title
            text += "Url:" + paper.url
            text += "Abstrat:" + paper.abs
            text += "Paper_info:" + paper.section_text_dict["paper_info"]
            # intro
            text += list(paper.section_text_dict.values())[0]

            try:
                chat_summary_text = self.chat_summary(text=text)
            except Exception as e:
                logger.info(f"summary_error: {e}")
                if "maximum context" in str(e):
                    current_tokens_index = (
                        str(e).find("your messages resulted in")
                        + len("your messages resulted in")
                        + 1
                    )
                    offset = int(
                        str(e)[current_tokens_index : current_tokens_index + 4]
                    )
                    summary_prompt_token = offset + 1000 + 150
                    chat_summary_text = self.chat_summary(
                        text=text, summary_prompt_token=summary_prompt_token
                    )

            htmls.append("## Paper:" + str(paper_index + 1))
            htmls.append("\n\n\n")
            chat_summary_text = ""
            if "chat_summary_text" in locals():
                htmls.append(chat_summary_text)

            # 第二步总结方法：
            # TODO，由于有些文章的方法章节名是算法名，所以简单的通过关键词来筛选，很难获取，后面需要用其他的方案去优化。
            method_key = ""
            for parse_key in paper.section_text_dict.keys():
                if "method" in parse_key.lower() or "approach" in parse_key.lower():
                    method_key = parse_key
                    break

            if method_key != "":
                text = ""
                method_text = ""
                summary_text = ""
                summary_text += "<summary>" + chat_summary_text
                # methods
                method_text += paper.section_text_dict[method_key]
                text = summary_text + "\n\n<Methods>:\n\n" + method_text
                # chat_method_text = self.chat_method(text=text)
                try:
                    chat_method_text = self.chat_method(text=text)
                except Exception as e:
                    logger.info(f"method_error: {e}")
                    if "maximum context" in str(e):
                        current_tokens_index = (
                            str(e).find("your messages resulted in")
                            + len("your messages resulted in")
                            + 1
                        )
                        offset = int(
                            str(e)[current_tokens_index : current_tokens_index + 4]
                        )
                        method_prompt_token = offset + 800 + 150
                        chat_method_text = self.chat_method(
                            text=text, method_prompt_token=method_prompt_token
                        )
                chat_method_text = ""
                if "chat_method_text" in locals():
                    htmls.append(chat_method_text)
                # htmls.append(chat_method_text)
            else:
                chat_method_text = ""
            htmls.append("\n" * 4)

            # 第三步总结全文，并打分：
            conclusion_key = ""
            for parse_key in paper.section_text_dict.keys():
                if "conclu" in parse_key.lower():
                    conclusion_key = parse_key
                    break

            text = ""
            conclusion_text = ""
            summary_text = ""
            summary_text += (
                "<summary>"
                + chat_summary_text
                + "\n <Method summary>:\n"
                + chat_method_text
            )
            if conclusion_key != "":
                # conclusion
                conclusion_text += paper.section_text_dict[conclusion_key]
                text = summary_text + "\n\n<Conclusion>:\n\n" + conclusion_text
            else:
                text = summary_text
            # chat_conclusion_text = self.chat_conclusion(text=text)
            try:
                chat_conclusion_text = self.chat_conclusion(text=text)
            except Exception as e:
                logger.info(f"conclusion_error: {e}")
                if "maximum context" in str(e):
                    current_tokens_index = (
                        str(e).find("your messages resulted in")
                        + len("your messages resulted in")
                        + 1
                    )
                    offset = int(
                        str(e)[current_tokens_index : current_tokens_index + 4]
                    )
                    conclusion_prompt_token = offset + 800 + 150
                    chat_conclusion_text = self.chat_conclusion(
                        text=text, conclusion_prompt_token=conclusion_prompt_token
                    )
            chat_conclusion_text = ""
            if "chat_conclusion_text" in locals():
                htmls.append(chat_conclusion_text)
            htmls.append("\n" * 4)

            # # 整合成一个文件，打包保存下来。
            date_str = str(datetime.datetime.now())[:13].replace(" ", "-")
            export_path = os.path.join(self.root_path, "export")
            if not os.path.exists(export_path):
                os.makedirs(export_path)
            mode = "w" if paper_index == 0 else "a"
            file_name = os.path.join(
                export_path,
                date_str
                + "-"
                + self.validateTitle(self.query)
                + "."
                + self.file_format,
            )
            self.export_to_markdown("\n".join(htmls), file_name=file_name, mode=mode)
            htmls = []

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    def chat_conclusion(self, text, conclusion_prompt_token=800):
        openai.api_key = self.chat_api_list[self.cur_api]
        self.cur_api += 1
        self.cur_api = (
            0 if self.cur_api >= len(self.chat_api_list) - 1 else self.cur_api
        )
        text_token = len(self.encoding.encode(text))
        clip_text_index = int(
            len(text) * (self.max_token_num - conclusion_prompt_token) / text_token
        )
        clip_text = text[:clip_text_index]

        messages = [
            {
                "role": "system",
                "content": "You are a reviewer in the field of ["
                + self.key_word
                + "] and you need to critically review this article",
            },
            # chatgpt 角色
            {
                "role": "assistant",
                "content": "This is the <summary> and <conclusion> part of an English literature, where <summary> you have already summarized, but <conclusion> part, I need your help to summarize the following questions:"
                + clip_text,
            },
            # 背景知识，可以参考OpenReview的审稿流程
            {
                "role": "user",
                "content": """
                 8. Make the following summary.Be sure to use {} answers (proper nouns need to be marked in English).
                    - (1):What is the significance of this piece of work?
                    - (2):Summarize the strengths and weaknesses of this article in three dimensions: innovation point, performance, and workload.
                    .......
                 Follow the format of the output later:
                 8. Conclusion: \n\n
                    - (1):xxx;\n
                    - (2):Innovation point: xxx; Performance: xxx; Workload: xxx;\n

                 Be sure to use {} answers (proper nouns need to be marked in English), statements as concise and academic as possible, do not repeat the content of the previous <summary>, the value of the use of the original numbers, be sure to strictly follow the format, the corresponding content output to xxx, in accordance with \n line feed, ....... means fill in according to the actual requirements, if not, you can not write.
                 """.format(
                    self.language, self.language
                ),
            },
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            # prompt需要用英语替换，少占用token。
            messages=messages,
        )
        result = ""
        for choice in response.choices:
            result += choice.message.content
        logger.info(f"conclusion_result:\n{result}")
        report_token_usage(response)

        return result

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    def chat_method(self, text, method_prompt_token=800):
        openai.api_key = self.chat_api_list[self.cur_api]
        self.cur_api += 1
        self.cur_api = (
            0 if self.cur_api >= len(self.chat_api_list) - 1 else self.cur_api
        )
        text_token = len(self.encoding.encode(text))
        clip_text_index = int(
            len(text) * (self.max_token_num - method_prompt_token) / text_token
        )
        clip_text = text[:clip_text_index]
        messages = [
            {
                "role": "system",
                "content": "You are a researcher in the field of ["
                + self.key_word
                + "] who is good at summarizing papers using concise statements",
            },
            # chatgpt 角色
            {
                "role": "assistant",
                "content": "This is the <summary> and <Method> part of an English document, where <summary> you have summarized, but the <Methods> part, I need your help to read and summarize the following questions."
                + clip_text,
            },
            # 背景知识
            {
                "role": "user",
                "content": """
                 7. Describe in detail the methodological idea of this article. Be sure to use {} answers (proper nouns need to be marked in English). For example, its steps are.
                    - (1):...
                    - (2):...
                    - (3):...
                    - .......
                 Follow the format of the output that follows:
                 7. Methods: \n\n
                    - (1):xxx;\n
                    - (2):xxx;\n
                    - (3):xxx;\n
                    ....... \n\n

                 Be sure to use {} answers (proper nouns need to be marked in English), statements as concise and academic as possible, do not repeat the content of the previous <summary>, the value of the use of the original numbers, be sure to strictly follow the format, the corresponding content output to xxx, in accordance with \n line feed, ....... means fill in according to the actual requirements, if not, you can not write.
                 """.format(
                    self.language, self.language
                ),
            },
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        result = ""
        for choice in response.choices:
            result += choice.message.content
        logger.info("method_result:\n", result)
        report_token_usage(response)

        return result

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    def chat_summary(self, text, summary_prompt_token=1100):
        openai.api_key = self.chat_api_list[self.cur_api]
        self.cur_api += 1
        self.cur_api = (
            0 if self.cur_api >= len(self.chat_api_list) - 1 else self.cur_api
        )
        text_token = len(self.encoding.encode(text))
        clip_text_index = int(
            len(text) * (self.max_token_num - summary_prompt_token) / text_token
        )
        clip_text = text[:clip_text_index]
        messages = [
            {
                "role": "system",
                "content": "You are a researcher in the field of ["
                + self.key_word
                + "] who is good at summarizing papers using concise statements",
            },
            {
                "role": "assistant",
                "content": "This is the title, author, link, abstract and introduction of an English document. I need your help to read and summarize the following questions: "
                + clip_text,
            },
            {
                "role": "user",
                "content": """
                 1. Mark the title of the paper (with Chinese translation)
                 2. list all the authors' names (use English)
                 3. mark the first author's affiliation (output {} translation only)
                 4. mark the keywords of this article (use English)
                 5. link to the paper, Github code link (if available, fill in Github:None if not)
                 6. summarize according to the following four points.Be sure to use {} answers (proper nouns need to be marked in English)
                    - (1):What is the research background of this article?
                    - (2):What are the past methods? What are the problems with them? Is the approach well motivated?
                    - (3):What is the research methodology proposed in this paper?
                    - (4):On what task and what performance is achieved by the methods in this paper? Can the performance support their goals?
                 Follow the format of the output that follows:
                 1. Title: xxx\n\n
                 2. Authors: xxx\n\n
                 3. Affiliation: xxx\n\n
                 4. Keywords: xxx\n\n
                 5. Urls: xxx or xxx , xxx \n\n
                 6. Summary: \n\n
                    - (1):xxx;\n
                    - (2):xxx;\n
                    - (3):xxx;\n
                    - (4):xxx.\n\n

                 Be sure to use {} answers (proper nouns need to be marked in English), statements as concise and academic as possible, do not have too much repetitive information, numerical values using the original numbers, be sure to strictly follow the format, the corresponding content output to xxx, in accordance with \n line feed.
                 """.format(
                    self.language, self.language, self.language
                ),
            },
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )

        result = ""
        for choice in response.choices:
            result += choice.message.content
        logger.info(f"summary_result:\n{result}")

        report_token_usage(response)

        return result

    def export_to_markdown(self, text, file_name, mode="w"):
        # 打开一个文件，以写入模式
        with open(file_name, mode, encoding="utf-8") as f:
            # 将html格式的内容写入文件
            f.write(text)

    # 定义一个方法，打印出读者信息
    def show_info(self):
        logger.info(f"Key word: {self.key_word}")
        logger.info(f"Query: {self.query}")
        logger.info(f"Sort: {self.sort}")


def add_subcommnd(parser):
    name = "arxiv"
    subparser = parser.add_parser(name, help="Fetch and summary paper from arxiv")
    subparser.add_argument(
        "--query",
        type=str,
        default="GPT-4",
        metavar="",
        help="the query string, ti: xx, au: xx, all: xx,",
    )
    subparser.add_argument(
        "--key-word",
        type=str,
        default="GPT robot",
        metavar="",
        help="the key word of user research fields",
    )
    subparser.add_argument(
        "--page-num", type=int, default=1, metavar="", help="the maximum number of page"
    )
    subparser.add_argument(
        "--max-results",
        type=int,
        default=1,
        metavar="",
        help="the maximum number of results",
    )
    subparser.add_argument(
        "--days",
        type=int,
        default=1,
        metavar="",
        help="the last days of arxiv papers of this query",
    )
    subparser.add_argument(
        "--sort", type=str, default="web", metavar="", help="another is LastUpdatedDate"
    )
    subparser.add_argument(
        "--save-image",
        default=False,
        metavar="",
        help="save image? It takes a minute or two to save a picture! But pretty",
    )

    subparser.add_argument(
        "--file-format",
        type=str,
        default="md",
        metavar="",
        help="export file format, if you want to save pictures, md is the best, if not, txt will not be messy",
    )

    subparser.add_argument(
        "--language",
        type=str,
        default="en",
        metavar="",
        help="The other output lauguage is English, is en",
    )

    return name


def main(args):
    reader1 = Reader(key_word=args.key_word, query=args.query, args=args)
    reader1.show_info()
    paper_list = reader1.get_arxiv_web(
        args=args, page_num=args.page_num, days=args.days
    )

    reader1.summary_with_chat(paper_list=paper_list)


def cli(args):
    parameters = ArxivParams(**vars(args))
    main(parameters)
