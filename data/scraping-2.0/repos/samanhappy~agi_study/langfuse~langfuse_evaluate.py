from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langfuse import Langfuse
from langfuse.client import CreateScore
from tqdm import tqdm
import os

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


def bleu_score(output, expected_output):
    def _tokenize(sentence):
        # 正则表达式定义了要去除的标点符号
        return re.sub(r"[^\w\s]", "", sentence.lower()).split()

    return sentence_bleu(
        [_tokenize(expected_output)],
        _tokenize(output),
        smoothing_function=SmoothingFunction().method3,
    )


prompt_template = """
Answer user's question according to the context below. 
Be brief, answer in no more than 20 words.
CONTEXT_START
{context}
CONTEXT_END

USER QUESTION:
{input}
"""


# 定义语言模型
llm = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    temperature=0,
)

# 定义Prompt模板
prompt = PromptTemplate.from_template(prompt_template)

# 检索 wikipedia
retriever = WikipediaRetriever(top_k_results=1)


# 定义输出解析器
parser = StrOutputParser()

wiki_qa_chain = (
    {"context": retriever, "input": RunnablePassthrough()} | prompt | llm | parser
)

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

dataset = langfuse.get_dataset("wiki_qa-20")

for item in tqdm(dataset.items):
    handler = item.get_langchain_handler(run_name="test_wiki_qa-20")

    output = wiki_qa_chain.invoke(item.input, config={"callbacks": [handler]})

    handler.rootSpan.score(
        CreateScore(name="bleu_score", value=bleu_score(output, item.expected_output))
    )
