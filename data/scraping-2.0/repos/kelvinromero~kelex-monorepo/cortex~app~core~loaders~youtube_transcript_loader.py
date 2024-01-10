from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import ElasticsearchStore

transcript = YouTubeTranscriptApi.get_transcript('BcMm0aaqnnI')
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 20,
)

documents = text_splitter.create_documents([line['text'] for line in transcript])

for i, doc in enumerate(documents):
    doc.metadata["episode_id"] = "BcMm0aaqnnI"
    doc.metadata["start"] = transcript[i]['start']
    doc.metadata["duration"] = transcript[i]['duration']
    doc["id"] = f"{doc.metadata['episode_id']}_{doc.metadata['start']}"

embeddings = OpenAIEmbeddings()

vectorstore = ElasticsearchStore.from_documents(
    documents,
    embeddings,
    es_url="http://es:9200",
    index_name="transcripts-vector",
    strategy=ElasticsearchStore.ApproxRetrievalStrategy()
)

db = ElasticsearchStore(
    es_url="http://es:9200",
    index_name="transcripts-vector",
    embedding=embeddings,
    strategy=ElasticsearchStore.ApproxRetrievalStrategy()
)

result = db.similarity_search(
    query="prime",
    k=1,
    filter={"match": {"metadata.episode_id": "BcMm0aaqnnI"}}
)


template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(temperature=0)

chain = prompt | llm

context = "\n".join([r.page_content for r in result])

chain.invoke({
    "context": context,
    "question": "Which year is it"
})
