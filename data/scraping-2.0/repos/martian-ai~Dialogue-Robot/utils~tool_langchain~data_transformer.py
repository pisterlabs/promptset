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

from langchain.document_transformers import BeautifulSoupTransformer
from langchain.text_splitter import CharacterTextSplitter


def data_transformer_bs(html):
    """ 对html 网页进行解析

    Args:
        html (_type_): html 网页信息 【类型有待确认】

    Returns:
        _type_: 解析之后的结果 【类型有待确认】
    """
    bs_transformer = BeautifulSoupTransformer()

    docs_transformed = bs_transformer.transform_documents(html,tags_to_extract=["span"])
    return docs_transformed

def text_spliter(documents):
    """对文档进行切分

    Args:
        documents (_type_): _description_

    Returns:
        _type_:
    """
    text_spliter = CharacterTextSplitter(chunk_size=32, chunk_overlap=0)
    texts = text_spliter.split_documents(documents)
    for item in texts:
        print(">>>")
        print(type(item))
    # print(texts[0])
    # print(texts[-1])

    return texts