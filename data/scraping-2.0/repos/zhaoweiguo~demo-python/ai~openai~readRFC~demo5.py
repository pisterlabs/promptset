from langchain.prompts import PromptTemplate  
from langchain.llms import OpenAI  
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory

llm = OpenAI(temperature=0)  
memory = ConversationSummaryMemory(llm=OpenAI())  

prompt_template = """你是一个中国IT开发者,对RFC标准特别熟悉,用中文对各RFC文档中的各小节总结,然后再对整体进行总结。需要满足以下要求:
1. 你的回答必须是中文
2. 回答尽量在800-1000个汉字范围

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

# sys.setrecursionlimit(10000000)

embedding_encoding = "cl100k_base"
encoding = tiktoken.get_encoding(embedding_encoding)

maxtoken=2000
def get_token(c):
    return len(encoding.encode(c))


def do_action(file_name, v):
    reply = conversation_with_summary.predict(input=v)
    # reply = "$$$$$$$$$"
    with open(file_name, 'a') as f:
        f.write(reply)
        f.write("\n")

defaultSep = "\d+."
def split_content(num, subcontent, file_name, sep):
    with open(file_name, 'a') as f:
        f.write(f'{num}> '+subcontent.split("\n")[0])
        f.write("\n")
    if get_token(subcontent) > maxtoken:
        newSep=sep+defaultSep
        # print("-", sep, '——', defaultSep, "=", newSep)
        # print(f'-{newSep}_')
        small_chapters = re.split(newSep+" ", subcontent)  
        for j, small_chapter in enumerate(small_chapters):
            newNum = f'{num}{j}.'
            # print(f'^{newNum}+')
            return split_content(newNum, small_chapter, file_name, newSep)
    else:
        return do_action(file_name, subcontent) 

defaultDir='resources/rfc'
file_list = os.listdir(defaultDir)
print("==========================================")

for file_name in file_list:
    full_path = os.path.join(defaultDir, file_name)  # 获取文件的完整路径
    print(full_path)
    if os.path.isfile(full_path):
        with open(file_name, 'w') as f:
            f.write(file_name+"\n")

        with open(full_path) as f2:
            content = f2.read()
            chapters = re.split('\n\d+.'+" ", content)

            for i, chapter in enumerate(chapters):
                if i==0:
                    continue
                split_content(f'{i}.', chapter, file_name, "\n\d+.") 


print("==========================================")
exit










