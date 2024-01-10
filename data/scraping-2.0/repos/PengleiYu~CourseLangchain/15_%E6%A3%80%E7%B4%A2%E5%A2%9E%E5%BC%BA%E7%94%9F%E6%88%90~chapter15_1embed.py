# 测试嵌入过程
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings()
embeddings = embeddings_model.embed_documents(
    ["您好，有什么需要帮忙的吗？",
     "哦，你好！昨天我订的花几天送达",
     "请您提供一些订单号？",
     "12345678",
     ],
)
print(len(embeddings))
# 嵌入结果很长
print(len(embeddings[0]))

