from langchain.prompts import PromptTemplate  
from langchain.llms import OpenAI  
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory

llm = OpenAI(temperature=0)  
memory = ConversationSummaryMemory(llm=OpenAI())  

prompt_template = """你是一个中国IT开发者,对RFC标准特别熟悉,用中文对各RFC文档中的各小节总结,然后再对整体进行总结。需要满足以下要求:
1. 你的回答必须是中文

{history}  
Human: {input}
AI:"""
prompt = PromptTemplate(  
    input_variables=["history", "input"], template=prompt_template  
)
conversation_with_summary = ConversationChain(
    llm=llm,  
    memory=memory,
    prompt=prompt,
    verbose=True  
)

import re
import tiktoken
import sys
import os


embedding_encoding = "cl100k_base"
encoding = tiktoken.get_encoding(embedding_encoding)
def get_token(c):
    return len(encoding.encode(c))


defaultSplit = "\d+."
def split_content(num, subcontent, outputF, split):
    outputF.write(f'{num} '+subcontent.split("\n")[0]+"\n")
    get_token(subcontent) > 2000

defaultDir='ai/openai/rfc'
file_list = os.listdir(defaultDir)

for file_name in file_list:
    full_path = os.path.join(defaultDir, file_name)  # 获取文件的完整路径
    if os.path.isfile(full_path):
        outputF=open(file_name, 'w')
        outputF.write(file_name+"\n")

        with open(full_path) as f:
            content = f.read()
            chapters = re.split('\n\d+.', content)

            for i, chapter in enumerate(chapters):
                if i==0:
                    continue
                split_content(f'{i}', chapter, outputF, "\n\d+.") 








