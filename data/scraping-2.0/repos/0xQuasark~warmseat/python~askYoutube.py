import argparse
from dotenv import load_dotenv
load_dotenv()

from langchain.document_loaders import YoutubeLoader
from langchain.indexes import VectorstoreIndexCreator

# https://www.youtube.com/watch?v=_mkBlyBGTD8

def extract_youtube_id(youtube_url):
  return youtube_url.split('v=')[-1]

def load_and_vectorize(youtube_url_id):
  loader = YoutubeLoader(youtube_url_id, add_video_info=False)
  docs = loader.load()
  index = VectorstoreIndexCreator()
  return index.from_documents(docs)

def query_index(index, query):
  return index.query(query)


def main():
  parser = argparse.ArgumentParser(description='Query a Youtube Video: ')
  parser.add_argument('-url', type=str, help='URL of the Youtube Video')
  args = parser.parse_args()

  youtube_url_id = extract_youtube_id(args.url).split('=')[-1]
  index = load_and_vectorize(youtube_url_id)

  while True:
    query = input('What do you want to ask the video?')
    response = query_index(index, query)
    print(f"Answer: {response}")
    if query == 'quit' or query == 'q':
      break
        
if __name__ == '__main__':
  # url = 'https://www.youtube.com/watch?v=_mkBlyBGTD8'
  # print(extract_youtube_id(url))
  main()




