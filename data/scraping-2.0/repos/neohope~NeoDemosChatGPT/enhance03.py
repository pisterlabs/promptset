#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai
import yaml
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex
from llama_index.readers.file.base import DEFAULT_FILE_EXTRACTOR, ImageParser
from llama_index.response.notebook_utils import display_response, display_image
from llama_index.indices.query.query_transform.base import ImageOutputQueryTransform

'''
llama_index增强检索
先通过OCR识别图像中的信息
然后进行查询
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    get_api_key()

    # 初始化ImageParser
    image_parser = ImageParser(keep_image=True, parse_text=True)
    file_extractor = DEFAULT_FILE_EXTRACTOR
    file_extractor.update(
    {
        ".jpg": image_parser,
        ".png": image_parser,
        ".jpeg": image_parser,
    })

    # 加载图片
    filename_fn = lambda filename: {'file_name': filename}
    receipt_reader = SimpleDirectoryReader(
        input_dir='./data/receipts', 
        file_extractor=file_extractor, 
        file_metadata=filename_fn,
    )

    # 图片转换为文本
    receipt_documents = receipt_reader.load_data()
    
    # 文本转向量
    receipts_index = GPTSimpleVectorIndex.from_documents(receipt_documents)

    # 查询
    receipts_response = receipts_index.query(
        'When was the last time I went to McDonald\'s and how much did I spend. \
        Also show me the receipt from my visit.',
        query_transform=ImageOutputQueryTransform(width=400)
    )
    display_response(receipts_response)

    # 输出图片文本
    output_image = image_parser.parse_file('./data/receipts/1100-receipt.jpg')
    print(output_image.text)
