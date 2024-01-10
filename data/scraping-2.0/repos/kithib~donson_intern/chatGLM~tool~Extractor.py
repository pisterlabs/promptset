import sys
#print(sys.path)    # 添加前的环境变量列表
sys.path.append(r'/home/kit/kit/ChatGLM3/langchain_demo/Tool/es')    # 添加顶级目录
#print(sys.path)    # 添加后的环境变量列表
import os
import json
from typing import Any
from es import init_es , combine_search
import requests
from langchain.tools import BaseTool


class Extractor(BaseTool):  # 提取主要信息
    name = "Extractor"
    description = "Use ES to search the best n text"
    def __init__(self):
        super().__init__()
    #text输入为一个json行，load后为字典 data = {"brand": , "category": ,"feature": ,"crowd": }
    def get_TopnText(self, text):
        data = json.loads(text)
        brand = data["brand"]
        category = data["category"]
        feature = data["feature"]
        crowd = data["crowd"]
        query = {"brand":brand,"category":category,"feature":feature,"crowd":crowd}
        results = combine_search(query, topn=5,fields=['brand', 'category', 'feature', 'crowd', 'ctime', 'title', 'content'])
        return results
    def _run(self, para: str) -> str:
        return self.get_TopnText(para)


if __name__ == "__main__":
    # 初始化
    #init_es()
    extract_tool = Extractor()
    prompt1 = """{"category": "美容仪","brand": "TriPollar","feature": "修复肌肤,美白,抑痘,无刺痛感","crowd":"敏感肌群体"}"""
    extract_info = extract_tool.run(prompt1)
    print(extract_info)
