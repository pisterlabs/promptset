
import openai, os
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, QuestionAnswerPrompt

# llama-index 默认使用的模型是 text-davinci-003，价格比 gpt-3.5-turbo 要贵上十倍
openai.api_key = os.environ.get("OPENAI_API_KEY")

TARGET_PATH = './index_tyxs.json'

if os.path.exists(TARGET_PATH) == False:
    documents = SimpleDirectoryReader('./articles').load_data()
    # 它会把文档分段转换成一个个向量，然后存储成一个索引。
    index = GPTSimpleVectorIndex.from_documents(documents)
    # 把对应的索引存下来，存储的结果就是一个 json 文件。后面，我们就可以用这个索引来进行相应的问答。
    index.save_to_disk(TARGET_PATH)

index = GPTSimpleVectorIndex.load_from_disk(TARGET_PATH)

# 调用 Query 函数，就能够获得问题的答案
response = index.query("鲁迅先生在日本学习医学的老师是谁？")
print(response)

response = index.query("鲁迅先生是去哪里学的医学？", verbose=True)
print(response)


# 上述过程里面是如何提交给 OpenAI 的呢？
query_str = "鲁迅先生去哪里学的医学？"

# 模版里面支持两个变量，一个叫做 context_str，另一个叫做 query_str。context_str 的地方，在实际调用的时候，会被通过 Embedding 相似度找出来的内容填入。而 query_str 则是会被我们实际提的问题替换掉。
# 实际提问的时候，我们告诉 AI，只考虑上下文信息，而不要根据自己已经有的先验知识（prior knowledge）来回答问题。
DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {query_str}\n"
)
QA_PROMPT = QuestionAnswerPrompt(DEFAULT_TEXT_QA_PROMPT_TMPL)

response = index.query(query_str, text_qa_template=QA_PROMPT)
print(response)