from typing import Tuple, List
import json
from bertopic import BERTopic
# from langchain import OpenAI
from bertopic.representation import OpenAI
import numpy as np

def run_bertopic():
	file_path = f"data_store/embeddings_seed_69420_size_10000.json"
	embeddings: List[Tuple[str, List[float]]] = json.load(open(file_path, "r"))
	# docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
	# print(docs[0])
	docs = [e[0] for e in embeddings]
	embeddings = np.array([e[1] for e in embeddings])

	representation_model = OpenAI(model="gpt-3.5-turbo", chat=True)
	topic_model = BERTopic(representation_model=representation_model)
	# topic_model = BERTopic()
	topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
	# print("TOPICS", topics)
	print(topic_model.get_topic_info())
	topic_model.get_topic_info().to_csv("data_store/baseline_bertopic_topics.csv")

	# print(probs)

if __name__ == "__main__":
	run_bertopic()
# See:
# https://maartengr.github.io/BERTopic/index.html#quick-start
