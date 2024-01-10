import configparser
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

config = configparser.ConfigParser()
config.read('config.ini')
openai_api_key = config.get('api', 'openai_api_key')
openai_api_base = config.get('api', 'openai_api_base')



embedding = OpenAIEmbeddings(openai_api_key=openai_api_key, openai_api_base=openai_api_base)

#需要存入数据库的文本

# 1.“鹅膏菌具有巨大而雄伟的地上（地上）子实体（担子果）。”
# 2.“具有较大子实体的蘑菇是鹅膏菌。有些品种是全白色的。”
# 3.“鬼笔甲，又名死亡帽，是所有已知蘑菇中毒性最强的一种。”
texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
]

smallDB = Chroma.from_texts(texts, embedding)

# 问题：告诉我有关子实体大的全白蘑菇的信息
question = "Tell me about all-white mushrooms with large fruiting bodies"

docs = smallDB.similarity_search(question, k=2)
for doc in docs:
    print(doc.page_content)

print("---------------------")

docs = smallDB.max_marginal_relevance_search(question,k=2, fetch_k=3)
for doc in docs:
    print(doc.page_content)