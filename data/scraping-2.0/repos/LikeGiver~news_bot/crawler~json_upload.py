import anthropic
from pathlib import Path
import sys
# 将code_part的父目录添加到sys.path
sys.path.append(str(Path(__file__).parent.parent))

from code_part.vector_store import get_vector_store, get_embed_model
from code_part.text_process import split_text_file

from llama_hub.file.pymu_pdf.base import PyMuPDFReader
from llama_index.node_parser.text import SentenceSplitter
from llama_index.schema import TextNode
import json
from datetime import datetime
import re

def convert_to_datetime(s):
    # 使用正则表达式提取日期和时间
    match = re.search(r'(\d{4})年 (\d{1,2})月 (\d{1,2})日 (am|pm)(\d{1,2}):(\d{2})发布', s)
    if not match:
        raise ValueError("字符串格式不符合预期")

    year, month, day, period, hour, minute = match.groups()

    # 将12小时制的时间转换为24小时制
    hour = int(hour)
    if period.lower() == 'pm' and hour < 12:
        hour += 12
    elif period.lower() == 'am' and hour == 12:
        hour = 0

    # 构造标准的日期时间字符串
    standard_datetime_str = f"{year}-{month}-{day} {hour:02d}:{minute}"

    # 解析为datetime对象
    return datetime.strptime(standard_datetime_str, "%Y-%m-%d %H:%M")

file_path = "/home/ubuntu/data/tyk_code/news_bot/output_copy.json"

embed_model = get_embed_model()
vector_store = get_vector_store()

with open(file_path, 'r') as file:
    data = json.load(file)

metadata = {}
metadata['keys'] = ""

for element in data:
    text_original = element['html']
    if len(text_original) < 200:
        pass
    metadata = {}
    metadata['keys'] = "time: " + convert_to_datetime(element['time']).strftime("%Y-%m-%d %H:%M") + " url: " + element['real_url']
    print(metadata["keys"])
    text_chunks = split_text_file(text_original, max_chars_per_file=200, chunk_overlap=20, separator="\n")

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        print("chunk: ", idx)
        print(text_chunk)

        node = TextNode(
            text= metadata["keys"] + '\n\n' + str(text_chunk),
        )
        node.metadata = metadata if len(metadata) > 0 else "DEFAULT METADATA"
        nodes.append(node)
        
    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding
    
    vector_store.add(nodes)