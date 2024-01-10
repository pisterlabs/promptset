from langchain.chat_models import BedrockChat
from langchain.retrievers import WikipediaRetriever, RePhraseQueryRetriever
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

retriever = WikipediaRetriever(
    lang="ja",
    doc_content_chars_max=500,
)

llm_chain = LLMChain(
    llm=BedrockChat(
        model_id="anthropic.claude-v2",
    ),
    prompt=PromptTemplate(
        template="""以下の質問からWikipediaで検索するべきキーワードを抽出してください。
        質問： {question}
        """,
        input_variables=["question"]
    )
)

re_phrase_query_retriever = RePhraseQueryRetriever(
    llm_chain=llm_chain,
    retriever=retriever,
)

documents = re_phrase_query_retriever.get_relevant_documents(
    "私はラーメンが好きです。ところでバーボンウィスキーとは何ですか？")

print(documents)
