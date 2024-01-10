# -*- coding: utf-8 -*-
"""
参考：https://blog.csdn.net/weixin_41168304/article/details/121758203
"""
import gensim
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
import matplotlib.pyplot as plt
import matplotlib

# 准备数据
PATH = "shizi_qss.txt"  # 已经进行了分词的文档


def main():
    file_object2 = open(PATH, encoding='utf-8', errors='ignore').read().split('\n')
    data_set = []  # 建立存储分词的列表
    for i in range(len(file_object2)):
        result = []
        seg_list = file_object2[i].split()  # 读取每一行文本
        for w in seg_list:  # 读取每一行分词
            result.append(w)
        data_set.append(result)
    print(data_set)  # 输出所有分词列表

    dictionary = corpora.Dictionary(data_set)  # 构建 document-term matrix
    corpus = [dictionary.doc2bow(text) for text in data_set]  # 表示为第几个单词出现了几次
    Lda = gensim.models.ldamodel.LdaModel  # 创建LDA对象

    # # 计算困惑度
    # def perplexity(num_topics):
    #     ldamodel = Lda(corpus, num_topics=num_topics, id2word=dictionary, passes=100)  # passes为迭代次数，次数越多越精准
    #     print(ldamodel.print_topics(num_topics=num_topics, num_words=20))  # num_words为每个主题下的词语数量
    #     print(ldamodel.log_perplexity(corpus))
    #     return ldamodel.log_perplexity(corpus)
    #
    #
    # # 绘制困惑度折线图
    # x = range(1, 30)  # 主题范围数量
    # y = [perplexity(i) for i in x]
    # plt.plot(x, y)
    # plt.xlabel('主题数目')  # x坐标
    # plt.ylabel('困惑度大小')  # y坐标
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
    # matplotlib.rcParams['axes.unicode_minus'] = False  # 设置正常显示字符
    # plt.title('主题-困惑度变化情况')
    # plt.show()

    # 计算coherence
    def coherence(num_topics):
        ldamodel = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=100, random_state=1)
        print(ldamodel.print_topics(num_topics=num_topics, num_words=20))
        ldacm = CoherenceModel(model=ldamodel, texts=data_set, dictionary=dictionary, coherence='c_v')
        print(ldacm.get_coherence())
        return ldacm.get_coherence()

    # 绘制coherence折线图
    x = range(1, 30)
    y = [coherence(i) for i in x]
    plt.plot(x, y)
    plt.xlabel('主题数目')
    plt.ylabel('coherence大小')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.title('主题-coherence变化情况')
    plt.show()


if __name__ == '__main__':
    main()
