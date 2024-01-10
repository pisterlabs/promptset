from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant

from llama_index.readers import BeautifulSoupWebReader



def retreiving_text(urls_):
    documents = BeautifulSoupWebReader().load_data(urls=urls_)
    return documents



def text_summary(text_):
    chunks = [text_[i:i+1000] for i in range(0, len(text_), 1000)]
    result = '\n\n'.join(chunks)


    text_splitter = CharacterTextSplitter(
        separator = "\n\n",  # セパレータ
        chunk_size = 1000,  # チャンクの文字数
        chunk_overlap = 0,  # チャンクオーバーラップの文字数
    )

    split_texts = text_splitter.split_text(result)
    split_docs = [Document(page_content=t) for t in split_texts]


    template1 = """
    次の文章を日本語で300文字程度に要約してください．
    文章：{text}
    """


    template2 = """
    次の文章を日本語で1000文字程度に要約してください．
    文章：{text}
    """

    prompt1 = PromptTemplate(
        input_variables = ['text'],
        template = template1,
    )

    prompt2 = PromptTemplate(
        input_variables = ['text'],
        template = template2,
    )

    summary_llm = OpenAI(temperature=0, max_tokens = 1000)

    chain = load_summarize_chain(
        llm = summary_llm, 
        chain_type="map_reduce",
        # それぞれの要約を行うときのテンプレ
        map_prompt = prompt1,
        # 要約文の要約文を作るときのテンプレ
        combine_prompt = prompt2
        )

    summary = chain.run(split_docs)

    return summary



def create_vectordatabase(path = ''):
    embeddings = OpenAIEmbeddings()

    # qdrantのベクトルデータベースの作成
    sample = [Document(page_content="sample", metadata={"url": "sample" })]
    if path == '':
        qdrant = Qdrant.from_documents(sample, embeddings, location=":memory:", collection_name= "sample")
    else:
        qdrant = Qdrant.from_documents(sample, embeddings, path= path, collection_name="sample")

    return qdrant



def add_text(urls_, qdrant_):
    docs = retreiving_text(urls_)
    num_docs = len(docs)

    for d in range(num_docs):
        url = urls_[d]
        text = docs[d].text
        text = text.replace('\n', '')

        # 文章の要約
        summ_text = text_summary(text)

        # qdrantに登録
        docs_ = [Document(page_content=summ_text, metadata={"url": url})]
        qdrant_.add_documents(docs_, collection_name = "")
       
    return 