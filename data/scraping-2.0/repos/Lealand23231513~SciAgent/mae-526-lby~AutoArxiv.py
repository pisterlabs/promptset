import arxiv
import requests
import openai
import json
import os
from typing import Any, Dict, List, Optional, Union, ClassVar

from pydantic import BaseModel, root_validator
from langchain.schema import Document


class ArxivAPIWrapper(BaseModel):
    arxiv_exceptions:tuple = (
                arxiv.ArxivError,
                arxiv.UnexpectedEmptyPageError,
                arxiv.HTTPError,
            ) 
    arxiv_result:ClassVar = arxiv.Result
    top_k_results: int = 3
    ARXIV_MAX_QUERY_LENGTH: int = 300
    load_max_docs: int = 100
    load_all_available_meta: bool = False
    doc_content_chars_max: Optional[int] = 40000
    

    def run(self, query: str) -> list[dict[str, str]] | str:

        try:
            results = arxiv.Search(  
                query[: self.ARXIV_MAX_QUERY_LENGTH], max_results=self.top_k_results 
            ).results()
        except self.arxiv_exceptions as ex:
            return f"Arxiv exception: {ex}"
        docs = [
            {
                "Title": result.title,
                "arxiv_id": result.entry_id[21:],
                "summary": result.summary
            }
            for result in results
        ]

        if docs:
            return docs
        else:
            return "No good Arxiv Result was found"

def download_arxiv_pdf(arxiv_id, ori_name:str, folder_name:str):
    pdf_url = 'https://arxiv.org/pdf/' + arxiv_id + '.pdf'
    response = requests.get(pdf_url)
    if response.status_code == 200:
        # 定义一个字符映射表并创建翻译表，用来替换文件名中不能出现的字符
        char_mapping = {
            '\\': ' ',
            '/': ' ',
            '?': ' ',
            ':': ' ',
            '<': ' ',
            '>': ' ',
            '|': ' ',
            '*': ' ',
            '"': ' ',
        }
        translation_table = str.maketrans(char_mapping)

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        filename = os.path.join(folder_name.translate(translation_table), ori_name.translate(translation_table) + '.pdf')

        with open(filename, 'wb') as pdf_file:
            pdf_file.write(response.content)
        print(f'文件 {ori_name.translate(translation_table)}.pdf 下载成功！')
    else:
        print(f'下载失败，HTTP状态码: {response.status_code}')

    
    


def arxiv_auto_search_and_download(query:str, download:bool = True, top_k_results=3) -> list[dict[str, str]] | str | None:

    openai.api_key = os.environ["OPENAI_API_KEY"]

    prompt = f"""请你将该单词翻译成英文，并且只返回翻译后的英文单词：{query}"""
    messages = [{"role": "user", "content": prompt}] 
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )
    query = response.choices[0].message["content"]

    arxiv_wrapper = ArxivAPIWrapper(top_k_results=top_k_results)
    arxiv_result = arxiv_wrapper.run(f"""{query}""")

    if type(arxiv_result) == str:
        print(arxiv_result)
        return None
    
    
    print("get results:")
    for i,sub_dict in enumerate(arxiv_result):
        print(f"{i+1}.{json.dumps(sub_dict)}")

    # 如果不下载，直接返回
    if download == False:
        return arxiv_result
    
    folder_name=query
    same_name_cnt = 1

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        while(os.path.exists(folder_name+f"({str(same_name_cnt)})")):
            same_name_cnt += 1
        folder_name+=f"({str(same_name_cnt)})"
        print(f"当前路径下已存在同名文件夹，故创建新文件夹\"{folder_name}\"")
    
    path_entry = {
        "path":os.getcwd()+f"\\{folder_name}"
    }
 

    # 循环遍历result中的每个结果并下载论文
    for sub_dict in arxiv_result:
        if type(sub_dict) == dict:
            download_arxiv_pdf(sub_dict["arxiv_id"], sub_dict["Title"], folder_name)

    arxiv_result.append(path_entry)

    return arxiv_result

def search_and_download(user_input:str):
    messages = [{"role": "user", "content": f"{user_input}"}]
    with open('modules.json', "r") as f:
        module_descriptions = json.load(f)
    functions = module_descriptions[0]["functions"]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature = 0,
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]
    if response_message.get("function_call"):
        function_args = json.loads(response_message["function_call"]["arguments"])
        print(function_args)
        arg_download = function_args.get("download")
        arg_top_k_results = function_args.get("top_k_results")
        arxiv_result = arxiv_auto_search_and_download(query = function_args.get("query"),
                                                      download=arg_download if arg_download is not None else False,
                                                      top_k_results=arg_top_k_results if arg_top_k_results is not None else 3)
        return arxiv_result
    else:
        return None
