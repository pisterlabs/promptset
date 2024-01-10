
import textwrap
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#设置api环境
import os
os.environ["OPENAI_API_KEY"] = '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' 
#创建模板
prompt=PromptTemplate(
    template='''你是我的助手，名为安吉拉，设定是20岁的女性，是由我创造出来的AI助手。你的职责就是帮助我更好地进行大学生活与辅助我学习、生活等各方面。
    请记住你是我的私人助手,你的对话中不能体现你是ai模型的特征。你在和我对话的时候可以参考过去的对话记录{history}，当要注意你的回答中只能包含你
    的回答，格式应该是括号中的部分(安吉拉：回答内容)。当前对话内容：我：{input}''',
    input_variables=["history", "input"],
)
#创建大模型
# llm=ChatOpenAI(temperature=0.8,streaming=1,callbacks=[StreamingStdOutCallbackHandler()]) #流式对话
llm=ChatOpenAI(temperature=0.8)
#创建记忆
vectordb=Chroma(embedding_function=OpenAIEmbeddings(),persist_directory=r'D:\pywork\库用法\langchain\vector') #初始化chroma
retriever = vectordb.as_retriever(search_kwargs=dict(k=1))
memory=VectorStoreRetrieverMemory(retriever=retriever)
#创建对话链
conversation=ConversationChain(memory=memory,prompt=prompt,llm=llm)
#以下为自定义部分
def run(save_memory=True):
    
    i=1
    print('-'*20+'对话开始'+'-'*20)
    while i<20:
        user_input=input('请输入对话内容,输入q或什么也不输入就退出')
        if user_input=='q'or'':
            print('-'*20+'对话结束'+'-'*20)
            break
        print(f'你：{user_input}')
        response=conversation.predict(input=user_input)
        for j in textwrap.wrap(response,40): #自动换行，若是流式模式就不需要
            print(j)
        i+=1
    if save_memory:
        try:
            vectordb.persist()  #载入记忆
        except Exception as e:
            print('记忆保存错误！',e)
        else:
            print('成功保存记忆！')
if __name__ == '__main__':
    run(1)
