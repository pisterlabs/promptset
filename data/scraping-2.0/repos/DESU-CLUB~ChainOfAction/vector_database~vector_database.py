import os
import requests
import chromadb
import pandas as pd
from typing import Optional, Any, Iterable, List

from dotenv import load_dotenv
import pdb

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Could not import sentence_transformers: Please install sentence-transformers package.")
    
try:
    import chromadb
    from chromadb.api.types import EmbeddingFunction
except ImportError:
    raise ImportError("Could not import chromdb: Please install chromadb package.")
    
from typing import Dict, Optional, List
from rouge import Rouge

import openai

load_dotenv()

# Index knowledge base
# Load data
""" datasets = ['cs6101']
dataset = datasets[0]    # The current dataset to use
data_root = "data"
data_dir = os.path.join(data_root, dataset)
max_docs = -1
# print("Selected dataset:", dataset) """

def load_data_v1(data_dir, data_root):
    passages = pd.read_csv(os.path.join(data_dir, "leetcode.tsv"), sep='\t', header=0)
    # qas = pd.read_csv(os.path.join(data_dir, "questions.tsv"), sep='\t', header=0).rename(columns={"text": "question"})
    
    # We only use 5000 examples.  Comment the lines below to use the full dataset.
    passages = passages.head(5000)
    # qas = qas.head(5000)
    
    # return passages, qas
    return passages
# documents, questions = load_data_v1(data_dir, data_root)
""" documents = load_data_v1(data_dir, data_root)
documents['indextext'] = documents['title'].astype(str) + "\n" + documents['problem_text'] + "\n" + documents['skill_description']
 """
class MiniLML6V2EmbeddingFunction(EmbeddingFunction):
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    def __call__(self, texts):
        return MiniLML6V2EmbeddingFunction.MODEL.encode(texts).tolist()
emb_func = MiniLML6V2EmbeddingFunction()

# Set up Chroma upsert
class ChromaWithUpsert:
    def __init__(self, name,persist_directory, embedding_function,collection_metadata: Optional[Dict] = None,
    ):
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._embedding_function = embedding_function
        self._persist_directory = persist_directory
        self._name = name
        self._collection = self._client.get_or_create_collection(
            name=self._name,
            embedding_function=self._embedding_function
            if self._embedding_function is not None
            else None,
            metadata=collection_metadata,
        )

    def upsert_texts(
        self,
        texts: Iterable[str],
        metadata: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.
        Args:
            :param texts (Iterable[str]): Texts to add to the vectorstore.
            :param metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            :param ids (Optional[List[str]], optional): Optional list of IDs.
            :param metadata: Optional[List[dict]] - optional metadata (such as title, etc.)
        Returns:
            List[str]: List of IDs of the added texts.
        """
        # TODO: Handle the case where the user doesn't provide ids on the Collection
        if ids is None:
            import uuid
            ids = [str(uuid.uuid1()) for _ in texts]
        embeddings = None
        self._collection.upsert(
            metadatas=metadata, documents=texts, ids=ids
        )
        return ids

    def is_empty(self):
        return self._collection.count()==0

    def query(self, query_texts:str, n_results:int=5):
        """
        Returns the closests vector to the question vector
        :param query_texts: the question
        :param n_results: number of results to generate
        :return: the closest result to the given question
        """
        return self._collection.query(query_texts=query_texts, n_results=n_results)
    
""" # Embed and index documents with Chroma
chroma = ChromaWithUpsert(
    name=f"{dataset}_minilm6v2",
    embedding_function=emb_func,  # you can have something here using /embed endpoint
    persist_directory=data_dir,
)
if chroma.is_empty():
    _ = chroma.upsert_texts(
        texts=documents.indextext.tolist(),
        # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
        metadata=[{'id': id, 'title': title, 'problem_text': problem_text, 'skill_description': skill_description}
                  for (id, title, problem_text, skill_description) in
                  zip(documents.id, documents.title, documents.problem_text, documents.skill_description)],  # filter on these!
        ids=[str(i) for i in documents.id],  # unique for each doc
    )

# Select a question
# question_index = 65
# question_text = questions.question[question_index].strip("?") + "?"
question_text = "For a string x, find the length of the longest substring such that every character must be unique."
# print(question_text)

# Retrieve relevant context
relevant_chunks = chroma.query(
    query_texts=[question_text],
    n_results=5,
)
for i, chunk in enumerate(relevant_chunks['documents'][0]):
    print("=========")
    print("Paragraph index : ", relevant_chunks['ids'][0][i])
    print("Paragraph : ", chunk)
    print("Distance : ", relevant_chunks['distances'][0][i])

# Feed the context and the question to openai model
def make_prompt(context, question_text):
    return (f"{context}\n\nPlease answer a question using this "
            + f"text. "
            + f"If the question is unanswerable, say \"unanswerable\"."
            + f"Question: {question_text}")

context = "\n\n\n".join(relevant_chunks["documents"][0])
prompt = make_prompt(context, question_text)

# Insert context and question into openai model
openai.api_key = os.getenv("OPENAI_API_KEY")
response = openai.ChatCompletion.create(
  model = "gpt-4",
  messages = [{"role":"user",
              "content":f"{prompt}"}],  
  temperature=0.8,
  max_tokens=100,
  top_p=0.8,
  frequency_penalty=0,
  presence_penalty=0
)
print("Question = ", question_text)
print("Answer = ", response['choices'][0]['message']['content'])
# print("Expected Answer(s) (may not be appear with exact wording in the dataset) = ", questions.answers[question_index])


 """