import re
from typing import List
from bs4 import BeautifulSoup
import ast
from langchain.document_loaders import AsyncHtmlLoader
from agents.base_tools.base_tools import functional_Tool
from langchain.schema.language_model import BaseLanguageModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from configs.prompt_config import TOOLS_HTML_TOOL_PROMPT_TEMPLATE
import requests
from configs.base_config import GOOGLE_APIKEY, GOOGLE_SEARCH_ID
import json
from langchain.chains.summarize import load_summarize_chain
from langchain.schema.document import Document
from embeddings.embedding_helper import EmbeddingHelper


class Web_Parser_Tool_Response:
    original_infos: list = []
    content: str = None

    def __init__(self, content: str, original_infos: list) -> None:
        self.original_infos = original_infos
        self.content = content


class Web_Parser_Tool(functional_Tool):
    llm: BaseLanguageModel
    name = "内容搜索工具"
    description = "基于搜索引擎从互联网搜索相关内容的工具，输入搜索内容关键词，输出对应网页的内容"
    qa_template = TOOLS_HTML_TOOL_PROMPT_TEMPLATE
    prompt = PromptTemplate(input_variables=["query"], template=qa_template)
    llm_chain: LLMChain = None
    base_url = "https://www.googleapis.com/customsearch/v1"
    system_params = f"key={GOOGLE_APIKEY}&cx={GOOGLE_SEARCH_ID}"
    is_parser_detail: bool = True
    child_link_count: int = 30

    def parser_output(self, output: str):
        if not output:
            raise ValueError("未能获取到正确的输入信息")
        # 按换行符分割字符串
        lines = output.split('\n')
        keywords = lines[0].split('：')[1].strip()
        return keywords

    def _call_func(self, query) -> str:
        # 1、提取关键字
        if not query:
            raise ValueError("未能获取到正确的用户信息")
        self.get_llm_chain()
        # 通过用户输入识别出用户输入的关键字，用于到互联网平台搜索内容
        keywords = self.parser_output(self.llm_chain.predict(query=query))
        # searchType=image&start=1
        self.system_params = f"{self.system_params}&q={keywords}&lr=lang_zh-CN&sort=review-rating:d:s"
        search_result = json.loads(
            requests.get(url=f"{self.base_url}?{self.system_params}").text)
        content = []
        final_content = ""
        if search_result and "items" in search_result:
            if self.is_parser_detail is False:
                for item in search_result["items"][:2]:
                    content.append({
                        "title": item["title"],
                        "link": item["link"],
                        "snippet": item["snippet"]
                    })
            else:
                for item in search_result["items"][:2]:
                    url = item["link"]
                    detail = self._load_html_content(
                        url=url, query=query, display_link=item["displayLink"])
                    if not detail:
                        continue
                    content.append({
                        "title": item["title"],
                        "link": item["link"],
                        "detail": detail,
                        "snippet": item["snippet"]
                    })

                final_content = self._final_summary(content, query)
        return {
            "sumary_result":
            Web_Parser_Tool_Response(content=final_content,
                                     original_infos=content)
        }

    def _extract_links(self, html_content: str, display_link: str, query: str):
        links = []
        # 使用Beautiful Soup解析HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        # 查找所有的<a>标签
        for a_tag in soup.find_all('a', href=True):
            link = a_tag['href']
            text = a_tag.get_text(strip=True)
            # 相对路径且不是锚点时，进行链接组装
            if display_link not in link and "#" not in link:
                link = display_link + link
            if display_link in link and text:
                links.append({"link": link, "text": text})
        _prompt = """已知信息：'{context}'
        基于以上信息，请你筛选出哪些信息是与用户输入：'{query}'，关系较为密切的信息
        请严格按照如下格式输出————
        链接：[你认为与用户输入最密切的一条或多条信息，多条信息之间用逗号分隔] or []
        """
        context = ','.join(
            [item["text"] for item in links[:self.child_link_count]])

        prompt_template = PromptTemplate(input_variables=["context", "query"],
                                         template=_prompt)
        chain = LLMChain(llm=self.llm,
                         prompt=prompt_template,
                         llm_kwargs={
                             "temperature": 0.95,
                             "top_p": 0.7
                         })
        content = chain.predict(context=context, query=query)
        matches = (re.search(r'\[([^\]]+)\]', content)).group(0)
        result = json.loads(matches)
        matching_result = []
        if isinstance(result, list):
            matching_result = [
                item["link"] for item in links if item["text"] in result
            ]
        return matching_result

    def _final_summary(self, context_list: list, query: str):
        base_content_info = "\n\n".join(
            [obj["detail"] for obj in context_list])
        _prompt = """已知信息：'{content}'
        请以'{query}'为主题,完整详尽地归纳总结上述已知信息,与主题无关的内容可以忽略"""
        prompt_template = PromptTemplate(input_variables=["content", "query"],
                                         template=_prompt)
        chain = LLMChain(llm=self.llm,
                         prompt=prompt_template,
                         llm_kwargs={
                             "temperature": 0.95,
                             "top_p": 0.7
                         })
        content = chain.predict(content=base_content_info, query=query)
        return content

    def _is_could_as_input_response(self, web_content: str, query: str):

        def get_long_token_result(html_content: str):
            texts = EmbeddingHelper.splitText(html_content, 4000, 200)
            docs = [Document(page_content=text) for text in texts]
            prompt_temp = """对下面的文字做精简的摘要:{text}"""
            PROMPT = PromptTemplate(template=prompt_temp,
                                    input_variables=["text"])
            chain = load_summarize_chain(self.llm,
                                         chain_type="map_reduce",
                                         return_intermediate_steps=True,
                                         map_prompt=PROMPT,
                                         combine_prompt=PROMPT,
                                         verbose=True,
                                         llm_kwargs={
                                             "temperature": 0.8,
                                             "top_p": 0.7
                                         })

            summ = chain({"input_documents": docs}, return_only_outputs=True)
            return summ["output_text"]

        _prompt = """已知信息：'{content}'
        请你判断上述已知信息是否能作为'{query}'的回复内容，从而满足用户意图。
        如果能，请直接输出'能';
        如果不能，请直接输出'不能'。
        并且你需要按照规定格式进行输出，格式————“结论：能 or 不能”
        """
        prompt_template = PromptTemplate(input_variables=["content", "query"],
                                         template=_prompt)
        # 先总结内容，再让LLM判断是否能作为最终答案输出
        content = get_long_token_result(html_content=web_content)

        chain = LLMChain(llm=self.llm,
                         prompt=prompt_template,
                         llm_kwargs={
                             "temperature": 0.95,
                             "top_p": 0.7
                         })
        result = self.parser_output(chain.predict(content=content,
                                                  query=query))
        return content, True if result != "不能" else False

    def _load_html_content(self,
                           url: str,
                           query: str,
                           display_link: str = None):
        try:
            # 获取响应
            header = {
                "User-Agent":
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Accept-Language": "zh-CN,zh;q=0.9",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
            res = requests.get(url=url, headers=header).text
            # 使用正则表达式删除html和script
            content = Web_Parser_Tool._remove_html_and_js(res)

            content, is_result = self._is_could_as_input_response(
                web_content=content, query=query)
            if is_result is False:
                # 提取html中的链接地址
                link_list = self._extract_links(html_content=res,
                                                display_link=display_link,
                                                query=query)
                link_count = len(link_list)
                i = 0
                while i < link_count and i < self.child_link_count:
                    item_url = link_list[i]
                    i = i + 1
                    res = requests.get(url=item_url, headers=header).text
                    content = Web_Parser_Tool._remove_html_and_js(res)
                    content, is_result = self._is_could_as_input_response(
                        web_content=content, query=query)
                    if is_result is True:
                        break
        except:
            content = ""
            pass
        finally:
            return content

    @staticmethod
    def _remove_html_and_js(text):
        text = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '',
                      text)
        text = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', '',
                      text)
        text = re.sub(r'<.*?>', '', text)
        text = text.replace("\r", "").replace("\n", "")
        text = re.sub(r'<[^>]+>', '', text)
        return text

    @staticmethod
    def _remove_js_code(code):
        tree = ast.parse(code, mode='exec')
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.Call)):
                ast.fix_missing_locations(ast.parse('pass').body[0])
                ast.copy_location(ast.parse('pass').body[0], node)
        cleaned_code = ast.unparse(tree)
        return cleaned_code

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
