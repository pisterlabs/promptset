from logging import root
from os import write
import re
import time
import random
from turtle import mainloop
from change_sentence import *
from constant import *
from query_data import *
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import JSONLoader
from langchain.globals import set_verbose
import sys
from langchain.schema.output_parser import StrOutputParser
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
import pypandoc
import io
root_path = sys.argv[1]
set_verbose(True)
# 写作
main_llm = MAIN_LLM
question_generator_llm = QUESTION_GENERATOR_LLM
openai_llm = OPENAI_LLM
# convert ppt file  to text file
def get_random_graph(picture_name, content):
    def two_in_ten_chance():
        return random.random() < 0.2
    flag = two_in_ten_chance()
    if flag:
        chain = get_chain('graph', llm_name=openai_llm)
        chain.invoke({"input": content, "picture_name": picture_name})
    return flag
    
def get_thanks(content):
    thanks = """论文致谢:
    这篇论文的完成得到了许多人的帮助与支持。我首先要向我的指导老师表示最深的感激，感谢您的无私奉献和精心指导。同时，我也要感谢那些与我并肩奋斗的同学，你们的友情与支持是我前进的力量。最重要的，我要感谢我的父母，你们是我永远的坚强后盾。每当想到大学生活，我都会感慨万千。"""
    chain = get_chain('thanks', openai_llm)
    rt =  chain.invoke({"text": content, "thanks": thanks})
    rt = rt.replace("首先，", "").replace("其次，", "").replace("最后，", "").replace("再者，","").replace("再次，","")
    return rt 

def get_summary(content, title):
    chain = get_chain('summary', main_llm)
    rt =  chain.invoke({"content": content, "title": title})
    return rt
    
def get_abstract(content, title):
    chain = get_chain('abstract', llm_name=openai_llm)
    rt =  chain.invoke({"content": content, "title": title})
    return rt

def get_domestic_review(title, zh_refs):
    chain = get_chain('ref', openai_llm)
    out = chain.invoke({"reference": zh_refs, "domain": "国内", "title": title})
    out = handle_final_text(out)
    return out
    
def get_overseas_review(title, en_refs):
    chain = get_chain('ref', openai_llm)
    out = chain.invoke({"reference": en_refs, "domain": "国外", "title": title})
    out = handle_final_text(out)
    return out

def to_docx(filename):

# Read the Markdown file with UTF-8 encoding
    input_md = filename
    output_docx = input_md.replace('.md', '.docx')

    with io.open(input_md, 'r', encoding='utf-8') as md_file:
        md_content = md_file.read()

    # Convert the markdown content to docx
    pypandoc.convert_text(md_content, 'docx', format='md', outputfile=output_docx)
    print(f"Converted '{input_md}' to '{output_docx}' with Chinese content.")

def get_jsondb():
    print("first json")
    filename = "data/out/" + root_path + ".json"
    loader = JSONLoader(
        file_path=filename,
        jq_schema='.[].answer',
    )
    docs = loader.load()

    embedding = HuggingFaceInstructEmbeddings(
        model_name="../models/stella-large-zh-v2"
    )
    db = FAISS.from_documents(docs, embedding)
    print("Vectorestore created successful")
    return db

def write_main(question, db):
    print("Begin writing")
    _template = """
    <信息>{context}</信息>
    根据上面的信息, 写出{question}相关的内容, 只写一个段落, 作为论文的一部份.要具体的分析,有相关的作者,文献,数据就要提出来详细说明.
    请用一种连贯的叙述方式来阐述你的观点和论据。用丰富的词汇和多样的句式来表达你的观点,让你的回答更具可读性.句式和结构必须多变, 语句和词汇必须丰富。
    使用简体中文输出。
    """

    retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.15})

    info = retriever.get_relevant_documents(query=question)
    content='\n'.join([i.page_content for i in info])
    prompt = PromptTemplate.from_template(_template)
    llm = get_llm(main_llm)
    chain = prompt | llm | StrOutputParser()
    rt = chain.invoke({"context": content, "question": question})
    print(f"The first content is {rt}")
    return rt


def handle_final_text(text):
    # Remove the title
    pattern = re.compile(r'^标题.*\n', flags=re.MULTILINE)
    text = re.sub(pattern, '', text)
    # Replace all the bad words
    replace_lst = ["研究发现，","研究显示，","一些研究发现，","是的，","首先，","其次，","接下来，", "最后，"]
    for r in replace_lst:
        text = text.replace(r, '')
    return text

def revise_content(title, input):
    print("Begin revising content")
    chain = get_chain('revise', llm_name=main_llm) 
    rt = chain.invoke({"title": title, "input": input})
    rt = handle_final_text(rt)
    return rt

def extend_content(question, content):
    print("Begin extending")
    conclusions = ['总之', '总而言之', '结论如下','简而言之','概括地说','汇总来看','简言之','一言以蔽之','综合来看','回顾总结', '']
    conclusion = random.choice(conclusions)
    llm = get_llm(main_llm, temperature=0.3)
    temp = """
    原文:
    {text}

    对原文扩写, 作为{question}研究中的一部份，简体中文输出"""
    write_prompt = PromptTemplate.from_template(temp)
    write_chain = write_prompt | llm | StrOutputParser()
    rt = write_chain.invoke({"text":content, "question": question})
    rt = rt.replace("首先，", "").replace("其次，", "").replace("最后，", "").replace("此外，","").replace("再者，","").replace("再次，","").replace("综上所述，", '').replace("我们", "本文").replace("总的来说，", '').replace("研究者们","").replace("研究人员","").replace("研究表明，", "")
    pattern = r"（[^）]*）"
    rt = re.sub(pattern, "", rt)

    print(f"The second content is {rt}")
    return rt

def draft_to_revise(final_file, value, db):
    """vluse: 类似于雷电天气下民航飞机的飞行风险及预防措施分析中,研究背景，论文题目+小标题
    final_file: 写入的文件
    db: jsondb
    """
    sentence_change_chain = get_chain('sentence_change', llm_name=question_generator_llm)
    question = sentence_change_chain.invoke({"question": value})
    print(f"The quesiont is {question}")
    draft_text = write_main(question, db)
    revised_text = revise_content(value, draft_text)
    extended_text = extend_content(question, revised_text)
    final_file.write(extended_text + "\n\n")

def write_refs(title, final_file_path, zh_refs, en_refs):
    with open(final_file_path, "a+", encoding="utf-8") as f:
        f.seek(0)
        content = f.read()
        thanks = get_thanks(content[0:2000])
        zh_refs, en_refs = zh_refs.replace('\n', '\n\n'), en_refs.replace('\n', '\n\n')
        reference_part = "\n## 参考文献\n\n" + zh_refs + en_refs + '\n'
        f.write(reference_part)
        text = "\n## 致谢\n\n" + thanks + "\n"
        f.write(text)
        abstract_part = get_abstract(content[0:2000], title)
        en_abstract_part = get_chain('trans_en', llm_name=openai_llm).invoke({"context": abstract_part})
        f.write("\n## 摘要\n\n"+abstract_part + "\n\n")
        f.write("\n## Abstract\n\n"+en_abstract_part + "\n\n")

def main():
    """初稿=>修改=>扩展"""
    global root_path
    input_file = os.path.join('data', 'catalog', root_path +  '_catalog.md')
    with open(input_file, "r", encoding="utf-8") as f:
        catalog_dict = generate_statements(f.read())
    print(catalog_dict)
    title = list(catalog_dict.keys())[0]
    final_file_path = os.path.join( 'data', "final", root_path + "_final.md" )
    final_file = open(final_file_path, "w+", encoding="utf-8")
    db = get_jsondb()
    refs_path = os.path.join('data', 'refs', root_path + '_refs.md')
    with open(refs_path, 'r') as f:
        zh_refs, en_refs = f.read().split('---')
    for key, value in catalog_dict.items():
        final_file.write(key + "\n\n")
        if value:
            if '国内文献' in key or '国内研究' in key:
                print(f"key is {key}")
                print("这是国内文献")
                domestic_review = get_domestic_review(title, zh_refs)
                # revised_domestic_review = revise_content(value, domestic_review)
                final_file.write(domestic_review + "\n\n")
                continue
            if '国外文献' in key or '国外研究' in key:
                print(f"key is {key}")
                print("这是国外文献")
                overseas_review = get_overseas_review(title, en_refs)
                # revised_overseas_review = revise_content(value, overseas_review)
                final_file.write(overseas_review + "\n\n")
                continue
            if '总结' in key:
                print(f"key is {key}")
                final_file.seek(0)
                content = final_file.read()[0:4096]
                summary_part = get_summary(content, title)
                revised_summary_part = revise_content(value, summary_part)
                final_file.write(revised_summary_part + "\n\n")
                continue
            draft_to_revise(final_file, value, db)
    final_file.close()
    write_refs(title, final_file_path, zh_refs, en_refs)


    # to_docx(final_file_path)
    
    
if __name__ == "__main__":
    main()