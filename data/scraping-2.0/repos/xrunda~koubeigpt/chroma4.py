import os

from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
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
