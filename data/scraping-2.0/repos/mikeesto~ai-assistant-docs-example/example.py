import openai
import nltk
import pickle
import faiss
import numpy as np
from time import sleep
from pathlib import Path

# TODO: Add your key here
openai.api_key = ""

# TODO: Add your source text here. You can get it from anywhere - reading a file, using an API call etc.
source_text = "On Monday I climbed Mount Yari. On Tuesday I rested. On Wednesday I travelled by train to Hiroshima."


def chunk_text(text, chunk_size=1024):
  sentences = nltk.sent_tokenize(text)
  chunks = []
  chunk = ""
  for sentence in sentences:
    if len(chunk + " " + sentence) <= chunk_size:
      chunk = chunk + " " + sentence
    else:
      chunks.append(chunk)
      chunk = sentence
  chunks.append(chunk)
  return chunks


def get_embeddings(chunked_text):
  embeddings = []
  count = 0
  for chunk in chunked_text:
    count += 1
    embeddings.append(openai.Embedding.create(
        model="text-embedding-ada-002", input=chunk)["data"][0]["embedding"])
    # Prevent being rate limited by API (especially on free plans)
    if count % 30 == 0:
      sleep(60)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    pickle.dump(index, open('index.pickle', 'wb'))
    return index


def main():
  chunked_text = chunk_text(source_text)

  embeddings = None
  if not Path("index.pickle").exists():
    embeddings = get_embeddings(chunked_text)
  else:
    embeddings = pickle.load(open("index.pickle", "rb"))

  try:
    while True:
      question = input("Please ask your question: ")

      question_embedding = openai.Embedding.create(
          model="text-embedding-ada-002", input=question)["data"][0]["embedding"]
      _, indices = embeddings.search(np.array([question_embedding]), 4)

      relevant_text = []

      for i in indices[0]:
        if i == -1:
          break
        relevant_text.append(chunked_text[i])

      relevant_text = "\n".join(relevant_text)
      answer = openai.Completion.create(
          prompt=f"""Answer the question as truthfully as possible using the source text, and if the answer is not contained within the source text, say "I don't know". \n\n 
          Question: {question}
          Source text: {relevant_text}""",
          model="text-davinci-003"
      )

      print(answer["choices"][0]["text"].strip())

  except KeyboardInterrupt:
    print("\nExiting...")


if __name__ == '__main__':
  main()
