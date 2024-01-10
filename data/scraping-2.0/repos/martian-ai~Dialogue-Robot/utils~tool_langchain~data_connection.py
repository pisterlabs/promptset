'''
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2023 by Martain.AI, All Rights Reserved.
#
Description: # 
Author: # apollo2mars apollo2mars@gmail.com
################################################################################
'''

from langchain.document_loaders import UnstructuredHTMLLoader, BSHTMLLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import WebBaseLoader


def data_connection_load_html(path_html: str)-> list[str]:
    """加载本地的HTML 文件 【未测试通过】

    Args:
        path_html (str):HTML 文件的本地绝对地址

    Returns:
        list[str]: 返回的解析的文档
    """
    # loader = UnstructuredHTMLLoader(path_html)
    # docs = loader.load()
    # print(docs[0])

    loader = BSHTMLLoader(path_html)
    docs = loader.load()
    return docs


def data_connection_load_url(url_path:list[str])->list[str]:
    """针对url 网络地址的数据加载

    Args:
        url_path (list[str]): url 地址

    Returns:
        list[str]: 解析结果

    urls = [
        "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-8-2023",
        "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-9-2023"
    ]
    """
    loader = UnstructuredURLLoader(urls=url_path)
    data = loader.load()
    return data

def data_connection_load_web(web_path:str)->list[str]:
    """加载网络http地址的数据

    Args:
        web_path (str): http地址

    Returns:
        list[str]: 解析的文档
    """
    loader = WebBaseLoader(web_path) 
    data = loader.load()
    return data