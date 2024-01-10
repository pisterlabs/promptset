import os

from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

os.environ["OPENAI_API_KEY"]='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'

embeddings = OpenAIEmbeddings()


docs = [
    Document(
        page_content="一群科学家带回了恐龙，并引发了一场混乱",
        metadata={"year": 1993, "rating": 7.7, "genre": "科幻"},
    ),
    Document(
        page_content="建国初期，末代皇帝溥仪的一生，从皇帝到囚犯，从囚犯到普通人，从普通人到底层人民，从底层人民到底层人民",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="一步大棚在战争中故事，一群士兵保卫家园，最后被国家抛弃的悲壮故事。荣获奥斯卡最佳影片奖",
        metadata={"year": 2006, "director": "李光", "rating": 8.6},
    ),
    Document(
        page_content="一群身材苗条的女人非常健康，一些男人对她们念念不忘，青春期的男孩子们的梦想",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="玩具栩栩如生，令人目不暇接，一群玩具的故事，玩具的世界",
        metadata={"year": 1995, "genre": "动画", "rating": 8.8},
    ),
    Document(
        page_content="三个人走进无人区，寻找水源的故事",
        metadata={
            "year": 1979,
            "rating": 6.0,
            "director": "Andrei Tarkovsky",
            "genre": "科幻"
        },
    ),
]
vectorstore = Chroma.from_documents(docs, embeddings)


from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="电影类型",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="电影上映年份",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="导演",
        type="string",
    ),
    AttributeInfo(
        name="rating", 
        description="电影评分", 
        type="float"
    ),
]
document_content_description = "电影简介"
llm = OpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True,enable_limit=True
)

# This example only specifies a relevant query
print(retriever.get_relevant_documents("大棚导演的2000年之后上市的影片"))