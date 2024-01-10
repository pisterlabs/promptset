from langchain.chat_models import ChatOpenAI
from langchain.retrievers import WikipediaRetriever, RePhraseQueryRetriever
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

retriever = WikipediaRetriever(
    lang="ja",
    doc_content_chars_max=500
)

llm_chain = LLMChain(
    llm=ChatOpenAI(
        temperature=0
    ),
    prompt=PromptTemplate(
        input_variables=["question"],
        template="""以下の質問からWikipediaで検索するべきキーワードを抽出してください。
            質問: {question}
            """
    )
)

re_phrase_query_retriever = RePhraseQueryRetriever(
    llm_chain=llm_chain,
    retriever=retriever
)

documents = re_phrase_query_retriever.get_relevant_documents(
    "私はカレーライスが好きです。ところで、スコッチウイスキーとはなんですか？"
)

print(documents)