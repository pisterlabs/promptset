from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory

llm = OpenAI(temperature=0)
memory = ConversationSummaryMemory(llm=OpenAI())

prompt_template = """你是一个中国IT开发者，对RFC标准特别熟悉，用中文对各RFC文档中的各小节总结，然后再对整体进行总结。需要满足以下要求:
1. 你的回答必须是中文
2. 回答限制在300个字以内

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
    verbose=False
)




import re
import tiktoken

def get_token(content):
    embedding_encoding = "cl100k_base"
    encoding = tiktoken.get_encoding(embedding_encoding)
    return len(encoding.encode(content))

def do_action(file, content):
    with open(file, 'w') as f:
        reply = conversation_with_summary.predict(input=content)
        f.write(reply)


with open('resources/rfc/rfc7230.txt') as f:
    content = f.read()

chapters = re.split('\n\d+.', content)

for i, chapter in enumerate(chapters):
    tokens = get_token(chapter)
    print("=====> ",i, tokens)
    if tokens > 500:
        small_chapters = re.split('\n\d+\.\d+.', chapter)
        for j, small_chapter in enumerate(small_chapters):
            do_action(f'b{i+1}_{j+1}.txt', chapter)
    else:
        do_action(f'b{i+1}.txt', chapter)










