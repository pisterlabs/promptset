from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone  # 向量数据库
import os
from keys import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENV
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
import streamlit as st  # 网站创建
import gtts  # 文字转语音

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

directory_path = '.\\my_data'  # 文本数据文件所在的文件夹
pinecone_index_name = "tourism"  # ! put in the name of your pinecone index here


def make_html():
    # note 存储完成向量数据库之后，我们就可以运行下面的代码，用streamlit帮我们做一个简单的网页可以用来调用我们的机器人问答
    # App framework
    # 如何创建自己的网页机器人
    st.title('MyBot')  # 用streamlit app创建一个标题
    # 创建一个输入栏可以让用户去输入问题
    query = st.text_input('欢迎来到MyBot,你可以问我关于旅游的问题，例如：国内有什么小众旅游地推荐？')
    my_bar = st.progress(0, text='等待投喂问题哦') # 一个简单的美化，不重要

    # 开始搜索，解答
    if query:
        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_API_ENV
        )

        # llm是用来定义语言模型，在下面的例子，用的是openai，注意，此openai调用的是langchain方法不是openai本ai
        # temperature是表示需要输出结果保持稳定
        llm = OpenAI(temperature=0, max_tokens=-1, openai_api_key=OPENAI_API_KEY)
        print('1:' + str(llm) + '\n')
        my_bar.progress(10, text='正在查询新华字典')

        # embedding就是把文字变成数字，这个embedding显然也是要使用openai的，所以要调用openai的api
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        print('2: 构建OpenAI embedding: 其为' + str(embeddings))


        # 调用pinecone的数据库，给出数据集的名字，然后是给出我现在问题的embedding尝试找到数据库中相似的信息
        docsearch = Pinecone.from_existing_index("tourism", embedding=embeddings)
        print('3: 构建docsearch对象，其为' + str(docsearch))

        # 相似度搜索，例如疼678，痛679，搜索用户的问题的相似度，会回复一个k*500的embedding
        selected_chunks = docsearch.similarity_search(query, k=3)
        print('4: 在docsearch中查找，结果为:')
        for doc in selected_chunks:
            print(doc)
        my_bar.progress(60, text='找到点头绪了')


        # 调用langchain的load qa办法，他就是一个封装过了的从query到answer的调用方法，’stuff‘为一种放入openai的办法
        chain = load_qa_chain(llm, chain_type='stuff', verbose=True)
        print('5 构建出question answer chain: ' + str(chain) + '\n')
        my_bar.progress(90, text='可以开始生成答案了，脑细胞在燃烧')


        # 得到答案
        answer = chain.run(input_documents=selected_chunks, question=query, verbose=True)
        print('6:' + str(answer))
        my_bar.progress(100, text='好了')

        st.write(answer)
        audio = gtts.gTTS(answer, lang='zh')
        audio.save("audio.wav")
        st.audio('audio.wav', start_time=0)
        os.remove("audio.wav")




if __name__ == '__main__':
    make_html()

