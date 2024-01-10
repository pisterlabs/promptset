from argparse import ArgumentParser

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from common import ASSET_PATH, INDEX_NAME


if __name__ == '__main__':
  parser = ArgumentParser(description='Ask a question.')
  parser.add_argument('question', type=str, help='The question')
  args = parser.parse_args()

  faiss = FAISS.load_local(ASSET_PATH, OpenAIEmbeddings(), INDEX_NAME)
  chain = RetrievalQAWithSourcesChain.from_chain_type(llm=ChatOpenAI(temperature=0, model="gpt-4"), retriever=faiss.as_retriever())

  result = chain({"question": args.question})
  print(f"Answer: {result['answer']}")
  print(f"Sources: {result['sources']}")
