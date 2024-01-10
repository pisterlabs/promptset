import os

from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# from langchain.embeddings import ModelScopeEmbeddings

from langchain.vectorstores import Chroma
from langchain.llms import Tongyi


os.environ["DASHSCOPE_API_KEY"] = 'sk-38e455061c004036a70f661a768ba779'
DASHSCOPE_API_KEY='sk-38e455061c004036a70f661a768ba779'

os.environ["OPENAI_API_KEY"]='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'

embeddings = OpenAIEmbeddings()
# embeddings = ModelScopeEmbeddings(model_id="xrunda/m3e-base",model_revision="v1.0.4")


docs = [
    Document(
        page_content="优点:驾驶者之车，经典的50:50重量分配，带来无与伦比的操控体验。",
        metadata={"brand": '李光', "model": "L9", "name":"2.0手动","year": 2018,"price":214800},
    ),
    Document(
        page_content="缺点:三缸机启动震动大噪音大，纯电续航能力弱",
        metadata={"brand": '李光', "model": "L9", "name":"2.0手动", "year": 2019,"price":214800},
    ),
    Document(
        page_content="外观颜色包括赤橙黄绿青蓝紫，内饰颜色包括五彩斑斓的黑",
        metadata={"brand": '李光', "model": "L9", "name":"2.0手动", "year": 2019,"price":214800},
    ),
    Document(
        page_content="纯电模式下动力充沛，馈电状态下动力就很一般了，整体动力表现不太满意",
        metadata={"brand": '李光', "model": "L9", "name":"2.0T自动", "year": 2019,"price":214800},
    ),
    Document(
        page_content="高速费油",
        metadata={"brand": '李光', "model": "L9", "name":"2.0T自动", "year": 2019,"price":214800},
    ),
    Document(
        page_content="后期维护成本较高，基础保养要2000元，比30万的合资品牌保养要贵，这点不太满意",
        metadata={"brand": '李光', "model": "L9", "name":"2.0T自动", "year": 2019,"price":214800},
    ),
    Document(
        page_content="冰箱不错",
        metadata={"brand": '李光', "model": "L9", "name":"2.0T自动", "year": 2019,"price":214800},
    ),
    Document(
        page_content="隔音好",
        metadata={"brand": '李光', "model": "L9", "name":"2.0T自动", "year": 2019,"price":214800},
    ),
    Document(
        page_content="后备箱太小",
        metadata={"brand": '李光', "model": "L9", "name":"纯电动", "year": 2019,"price":214800},
    ),
    Document(
        page_content="灯光不够亮",
        metadata={"brand": '李光', "model": "L9", "name":"纯电动", "year": 2019,"price":214800},
    ),
    Document(
        page_content="底盘调教挺舒服的",
        metadata={"brand": '李光', "model": "L9", "name":"纯电动", "year": 2019,"price":214800},
    ),
    Document(
        page_content="座椅加热很舒服",
        metadata={"brand": '李光', "model": "L9", "name":"纯电动", "year": 2019,"price":214800},
    ),
    Document(
        page_content="根据我2万公里的形式历程来看，L9的城市综合油耗是9.5升，高速油耗是8.5升",
        metadata={"brand": '李光', "model": "L9", "year": 2019,"price":214800},
    ),
    Document(
        page_content="根据我2万公里的形式历程来看，L9的城市综合油耗是9.5升，高速油耗是8.5升",
        metadata={"brand": '李光', "model": "L8", "year": 2019,"price":214800},
    ),
    Document(
        page_content="发动机与变速箱匹配的好，没有什么顿挫感。方向盘力度轻盈，容易上手，女士开也没有问题。大众品牌保值率高，朗逸虽是启航版，开起来也不错，性价比之选。地盘很扎实，整体性很强，虽然有点路噪和风噪，但是都在可接受范围之内，毕竟价格不高。",
        metadata={"brand": '大众', "model": "朗逸", "year": 2019,"price":112900,"rating":"优点"},
    ),
    Document(
        page_content="优点：新一代卡罗拉总算是不再油腻了，之前的充满了油腻感。整体线条感很好，前脸和凯美瑞很像。也算上是家族式设计。",
        metadata={"brand": '丰田', "model": "卡罗拉", "year": 2020,"price":128800,"rating":"外观"},
    ),
    Document(
        page_content="缺点:太费油了",
        metadata={"brand": '丰田', "model": "卡罗拉", "year": 2020,"price":128800,"rating":"外观"},
    ),
    Document(
        page_content="优点:满意的那太多了 排气这一代有很大提升 15段阻尼可调节 HUD抬头显示 哈曼卡顿音响 原地增压弹射起步 L2级别的自动驾驶 虽说这车要这个没有用 但是不能没有",
        metadata={"brand": '大众', "model": "高尔夫", "year": 2021,"price":229800},
    ),
    Document(
        page_content="缺点:烧机油，二手车不保值，配件价格贵",
        metadata={"brand": '大众', "model": "高尔夫", "year": 2021,"price":229800},
    ),
    Document(
        page_content="最满意，肯定是外观和动力，加上红旗品牌，毕竟是国货之光。上一辆车是昂克赛拉，动力太弱空间太小，异响太多所以才换车的，配置拉满确实6，主动刹车好评，启动过两次，太吓人了",
        metadata={"brand": '红旗', "model": "红旗HS5", "year": 2022,"price":213800,"rating":"优点"},
    ),
    Document(
        page_content="非要说最不满意的话，车的品质是无可挑剔的，价格偏高，但是可以承受！最主要是1.5卡罗拉的活动不是太多，期待厂家或车商组织一些更好的活动，把车友们组织起来！",
        metadata={"brand": '丰田', "model": "卡罗拉", "year": 2021,"price":132800,"rating":"缺点"},
    ),
]
vectorstore = Chroma.from_documents(docs, embeddings)


from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="brand",
        description="汽车品牌",
        type="string",
    ),
    AttributeInfo(
        name="model",
        description="车型",
        type="string",
    ),
    AttributeInfo(
        name="name",
        description="具体车型名称",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="上市年份",
        type="integer",
    ),
    AttributeInfo(
        name="price", 
        description="售价", 
        type="integer"
    ),
    AttributeInfo(
        name="rating", 
        description="评价内容", 
        type="string"
    ),
]
document_content_description = "汽车车型的用户评价"
llm = OpenAI(temperature=0)
# llm = Tongyi(model_kwargs={"api_key":DASHSCOPE_API_KEY},model_name= "qwen-7b-chat-v1")

retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True
)

#✔️ 增加name属性
# print(retriever.get_relevant_documents(query="李光L9,2.0T自动优点"))

# filter 2.0T自动 丢失
# print(retriever.get_relevant_documents(query="我最近考虑买车，目前关注李光L9这款车，请介绍一下2.0T自动优点和缺点"))

# 这个可以，三个过滤条件
# print(retriever.get_relevant_documents(query="请介绍李光L9纯电动,这款车的缺点"))
# 四个过滤条件就不行了，目前最多只能三个过滤条件??????结论不扎实，纯电动这个过滤条件丢失了
# print(retriever.get_relevant_documents(query="请介绍李光L9纯电动,这款车的缺点"))


# ✔️ 可以找出缺点
# print(retriever.get_relevant_documents(query="李光L9的缺点"))


# ✔️ 全部找出来，把优点排前面，缺点排后面
# print(retriever.get_relevant_documents(query="丰田卡罗拉优点,2020年上市"))

# print(retriever.get_relevant_documents(query="驾驶者之车",metadata={"brand": '理想'}))

# This example only specifies a relevant query
# ✔️
# print(retriever.get_relevant_documents("大众高尔夫的优点"))
# ✔️ 
# print(retriever.get_relevant_documents("2020年之后上市的宝马"))
# print(retriever.get_relevant_documents("2015年之后上市的宝马"))

# 2.检索生成结果
def retrieve_info(query):
    return retriever.get_relevant_documents(query=query)

# 3.设置LLMChain和提示
llm=ChatOpenAI(temperature=0, model='gpt-3.5-turbo-16k-0613')
template = """
    你是一名掌握了全部汽车用户真实使用评价内容的智能回复机器人。
    我将发送给你一位客户关于汽车使用、购买建议、与其他品牌车型对比等方面的问题。
    客户希望你在真实车主评价的基础上，归纳总结形成一句结论性的内容。
    你应该根据真实车主的评价内容，总结归纳有用的信息形成回复内容，发送给这位客户，并遵循以下所有规则。
    1/ 在篇幅、语气、逻辑论证和其他细节方面，答复应与尽可能的专业，如实的表达问题的答案，不要增加你自己的幻觉。
    2/ 如果在真实车主评价内容中没有这个问题的相关答案，请回答：“很抱歉，基于真实车主的口碑数据，我暂时不能给出您这个问题的答案。“
    {message}
    以下是针对这个问题，真实车主评价内容：
    {best_practice}
    请为这个客户返回最符合问题的最佳回复内容：

    所有回复均为中文
"""
prompt=PromptTemplate(
    input_variables=["message","best_practice"],
    template=template
)
chain=LLMChain(llm=llm,prompt=prompt)
# 4.检索生成结果
def generate_response(message):
    best_practice = retrieve_info(message)
    print('message：',message)
    print('向量召回内容：',best_practice)
    response = chain.run(message=message,best_practice=best_practice)
    return response

message='李光L9的优点'
print(generate_response(message))