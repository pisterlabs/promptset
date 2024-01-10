##用jupyter notebook写的 先直接放上来了（（
pip install langchain
pip install arxiv
pip install pymupdf
pip install openai
import os
os.environ["OPENAI_API_KEY"] = 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import arxiv
def PaperGet():
  print("开始查找论文"+topic)
  search = arxiv.Search(
  query = topic,
  max_results = 10,
  sort_by = arxiv.SortCriterion.Relevance
)
  for result in search.results():
    paper_url=paper_url+result.entry_id
    print(result.title)
    print(result.entry_id)  
def ResearchStatus():
    llm=OpenAI(temperature=0.0)
    print(paper_url)
    prompt=PromptTemplate(
        input_variables=["topic"],
        template="You are an authoritative writer in the field of {topic}, \
        and you are reading several papers' abstracts , the urls of the papers are here:\
        {paper_url} .Please write an overview, about 500 word, of the current research situation in the field\
        according to the abstracts, and it should be concise, clear and easy to understand, in line with the original meaning,\
        and at the same time needs to include the history and current situation of the research on this topic, \
        the main theoretical views and techniques at this stage, the main direction of the topic, \
        the main problems need to be solved urgently and the main development trends. "
    )
    chain=LLMChain(llm=llm,prompt=prompt)
    print("I'm working now!")
    result1=chain.run(topic)
    return result1
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
 )
def ChineseTranslation(TOPIC):
    chat = ChatOpenAI(temperature=0)
    template="You are a helpful assistant that translates {input_language} to {output_language}. For the already English part in\
              input, you should not change it.For example, if you get an input like 'AI数据结构'，you should output like this\
              'AI data structure'."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    topic=chat(chat_prompt.format_prompt(input_language="Chinese", output_language="English", text=TOPIC).to_messages()).content
    print(topic)
    return topic
def EnglishTranslation(TOPIC):
    chat = ChatOpenAI(temperature=0)
    template="You are a helpful assistant that translates {input_language} to {output_language}.."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    topic=chat(chat_prompt.format_prompt(input_language="English", output_language="Chinese", text=TOPIC).to_messages()).content
    print(topic)
    return topic
topic=input("输入需要查找的论文领域：")
print("我明白了。你需要查找的论文领域是"+topic)
topic=ChineseTranslation(topic)
print("翻译成了英文"+topic)
PaperGet()
result=ResearchStatus()
result=EnglishTranslation(result)
print(result)
print("工作结束了!")
