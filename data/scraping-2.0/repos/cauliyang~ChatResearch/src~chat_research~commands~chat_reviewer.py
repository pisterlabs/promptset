import datetime
import os
import re
from pathlib import Path
from typing import Optional

import openai
import tenacity
import tiktoken
from loguru import logger
from pydantic import BaseModel, validator

from chat_research.utils import report_token_usage

from ..paper import Paper
from ..utils import load_config


class ReviewerParams(BaseModel):
    paper_path: str
    file_format: str
    review_format: Optional[str] = None
    research_fields: str
    language: str

    @validator("paper_path")
    def paper_path_must_exist(cls, v):
        if not Path(v).exists():
            raise ValueError("paper_path must exist")


REVIEW_FORMAT = """
* Overall Review
Please briefly summarize the main points and contributions of this paper.
xxx

* Paper Strength
Please provide a list of the strengths of this paper, including but not limited to: innovative and practical methodology, insightful empirical findings or in-depth theoretical analysis, well-structured review of relevant literature, and any other factors that may make the paper valuable to readers. (Maximum length: 2,000 characters)
(1) xxx
(2) xxx
(3) xxx
...

* Paper Weakness
Please provide a numbered list of your main concerns regarding this paper (so authors could respond to the concerns individually). These may include, but are not limited to: inadequate implementation details for reproducing the study, limited evaluation and ablation studies for the proposed method, correctness of the theoretical analysis or experimental results, lack of comparisons or discussions with widely-known baselines in the field, lack of clarity in exposition, or any other factors that may impede the reader's understanding or benefit from the paper. Please kindly refrain from providing a general assessment of the paper's novelty without providing detailed explanations. (Maximum length: 2,000 characters)
(1) xxx
(2) xxx
(3) xxx
...

* Questions To Authors And Suggestions For Rebuttal
Please provide a numbered list of specific and clear questions that pertain to the details of the proposed method, evaluation setting, or additional results that would aid in supporting the authors' claims. The questions should be formulated in a manner that, after the authors have answered them during the rebuttal, it would enable a more thorough assessment of the paper's quality. (Maximum length: 2,000 characters)
xxx

*Overall score (1-10)
The paper is scored on a scale of 1-10, with 10 being the full mark, and 6 stands for borderline accept. Then give the reason for your rating.
xxx
"""


class Reviewer:
    # 初始化方法，设置属性
    def __init__(self, args=None):
        if args is None:
            raise ValueError("args is None")

        if args.language == "en":
            self.language = "English"
        elif args.language == "zh":
            self.language = "Chinese"
        else:
            self.language = "English"

        self.research_fields = args.research_fields
        self.review_format = args.review_format

        self.config, self.chat_api_list = load_config()

        self.cur_api = 0
        self.file_format = args.file_format
        self.max_token_num = 4096
        self.encoding = tiktoken.get_encoding("gpt2")

    @staticmethod
    def get_review_format(path: Optional[Path]):
        if path is None:
            return REVIEW_FORMAT
        else:
            with open(path, "r") as f:
                return f.read()

    def validateTitle(self, title):
        # 修正论文的路径格式
        rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
        new_title = re.sub(rstr, "_", title)  # 替换为下划线
        return new_title

    def review_by_chatgpt(self, paper_list):
        htmls = []
        for paper_index, paper in enumerate(paper_list):
            sections_of_interest = self.stage_1(paper)
            # extract the essential parts of the paper
            text = ""
            text += "Title:" + paper.title + ". "
            text += "Abstract: " + paper.section_texts["Abstract"]
            intro_title = next(
                (item for item in paper.section_names if "ntroduction" in item.lower()),
                None,
            )
            if intro_title is not None:
                text += "Introduction: " + paper.section_texts[intro_title]
            # Similar for conclusion section
            conclusion_title = next(
                (item for item in paper.section_names if "onclusion" in item), None
            )
            if conclusion_title is not None:
                text += "Conclusion: " + paper.section_texts[conclusion_title]
            for heading in sections_of_interest:
                if heading in paper.section_names:
                    text += heading + ": " + paper.section_texts[heading]
            chat_review_text = self.chat_review(text=text)
            htmls.append("## Paper:" + str(paper_index + 1))
            htmls.append("\n\n\n")
            htmls.append(chat_review_text)

            # 将审稿意见保存起来
            date_str = str(datetime.datetime.now())[:13].replace(" ", "-")
            export_path = os.path.join(self.root_path, "export")
            if not os.path.exists(export_path):
                os.makedirs(export_path)
            mode = "w" if paper_index == 0 else "a"
            file_name = os.path.join(
                export_path,
                date_str
                + "-"
                + self.validateTitle(paper.title)
                + "."
                + self.file_format,
            )
            self.export_to_markdown("\n".join(htmls), file_name=file_name, mode=mode)
            htmls = []

    def stage_1(self, paper):
        text = ""
        text += "Title: " + paper.title + ". "
        text += "Abstract: " + paper.section_texts["Abstract"]
        openai.api_key = self.chat_api_list[self.cur_api]
        self.cur_api += 1
        self.cur_api = (
            0 if self.cur_api >= len(self.chat_api_list) - 1 else self.cur_api
        )
        messages = [
            {
                "role": "system",
                "content": f"You are a professional reviewer in the field of {self.research_fields}. "
                f"I will give you a paper. You need to review this paper and discuss the novelty and originality of ideas, correctness, clarity, the significance of results, potential impact and quality of the presentation. "
                f"Due to the length limitations, I am only allowed to provide you the abstract, introduction, conclusion and at most two sections of this paper."
                f"Now I will give you the title and abstract and the headings of potential sections. "
                f"You need to reply at most two headings. Then I will further provide you the full information, includes aforementioned sections and at most two sections you called for.\n\n"
                f"Title: {paper.title}\n\n"
                f"Abstract: {paper.section_texts['Abstract']}\n\n"
                f"Potential Sections: {paper.section_names[2:-1]}\n\n"
                f"Follow the following format to output your choice of sections:"
                f"{{chosen section 1}}, {{chosen section 2}}\n\n",
            },
            {"role": "user", "content": text},
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        result = ""
        for choice in response.choices:
            result += choice.message.content
        logger.info(result)
        return result.split(",")

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    def chat_review(self, text):
        openai.api_key = self.chat_api_list[self.cur_api]
        self.cur_api += 1
        self.cur_api = (
            0 if self.cur_api >= len(self.chat_api_list) - 1 else self.cur_api
        )
        review_prompt_token = 1000
        text_token = len(self.encoding.encode(text))
        input_text_index = int(
            len(text) * (self.max_token_num - review_prompt_token) / text_token
        )
        input_text = "This is the paper for your review:" + text[:input_text_index]

        review_format = self.get_review_format(self.review_format)

        messages = [
            {
                "role": "system",
                "content": "You are a professional reviewer in the field of "
                + self.research_fields
                + ". Now I will give you a paper. You need to give a complete review opinion according to the following requirements and format:"
                + review_format
                + " Please answer in {}.".format(self.language),
            },
            {"role": "user", "content": input_text},
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        result = ""
        for choice in response.choices:
            result += choice.message.content
        logger.info("********" * 10)
        logger.info(result)
        logger.info("********" * 10)

        report_token_usage(response)

        return result

    def export_to_markdown(self, text, file_name, mode="w"):
        # 使用markdown模块的convert方法，将文本转换为html格式
        # html = markdown.markdown(text)
        # 打开一个文件，以写入模式
        with open(file_name, mode, encoding="utf-8") as f:
            # 将html格式的内容写入文件
            f.write(text)


def add_subcommand(parser):
    name = "reviewer"
    subparser = parser.add_parser(name, help="Summary paper")
    subparser.add_argument(
        "-p",
        "--paper-path",
        type=str,
        metavar="",
        help="path of papers",
        required=True,
    )

    subparser.add_argument(
        "-f",
        "--file-format",
        type=str,
        default="md",
        choices=["txt", "md"],
        metavar="",
        help="output file format (default: %(default)s)",
    )

    subparser.add_argument(
        "-r", "--review-format", type=str, metavar="", help="review format"
    )

    subparser.add_argument(
        "--research-fields",
        type=str,
        default="computer science, artificial intelligence and reinforcement learning",
        metavar="",
        help="the research fields of paper (default: %(default)s)",
    )

    subparser.add_argument(
        "-l",
        "--language",
        type=str,
        default="en",
        metavar="",
        help="output language, en or zh (default: %(default)s)",
    )

    return name


def main(args):
    reviewer1 = Reviewer(args=args)
    paper_list = []

    if args.paper_path.endswith(".pdf"):
        paper_list.append(Paper(path=args.paper_path))
    else:
        for root, dirs, files in os.walk(args.paper_path):
            logger.info(f"root: {root}, dirs: {dirs}, files: {files}")
            for filename in files:
                # 如果找到PDF文件，则将其复制到目标文件夹中
                if filename.endswith(".pdf"):
                    paper_list.append(Paper(path=os.path.join(root, filename)))
    logger.info(
        "------------------paper_num: {}------------------".format(len(paper_list))
    )

    for paper_index, paper_name in enumerate(paper_list):
        paper = paper_name.path.split("\\")[-1]
        logger.info(f"{paper_index}, {paper}")

    reviewer1.review_by_chatgpt(paper_list=paper_list)


def cli(args):
    parameters = ReviewerParams(**vars(args))
    main(parameters)
