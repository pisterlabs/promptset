from abc import ABC, abstractmethod
from dataset.dataset import Dataset
from openai.embeddings_utils import get_embedding
import numpy as np
import math

def cosine_similarity(a, b):
	return np.dot(a, b) / (math.sqrt(np.dot(a, a)) * math.sqrt(np.dot(a, a)))

class Agent(ABC):
	@abstractmethod
	def input_initial_state(self, initial_state: str, knowledge_yaml: str) -> None:
		pass

	@abstractmethod
	def input_state_change(self, state_change: str) -> None:
		pass

	@abstractmethod
	def answer_query(self, query: str) -> str:
		pass

class DummyAgent(Agent):
	def input_initial_state(self, initial_state: str) -> None:
		pass

	def input_state_change(self, state_change: str) -> None:
		pass

	def answer_query(self, query: str) -> str:
		return ""

class Simulation:
	def __init__(self, dataset: Dataset, agent: Agent) -> None:
		self.dataset = dataset
		self.agent = agent
	
	def run(self):
		print(f"Initial State:\n{self.dataset.initial_state}")
		self.agent.input_initial_state(self.dataset.initial_state, self.dataset.initial_knowledge_yaml)
		for time_step in self.dataset:
			print("Time: " + str(time_step["time"]))
			if time_step["type"] == "state change":
				print("State change: {}\n".format(time_step["state change"]))
				self.agent.input_state_change(time_step["state change"])
			else:
				print("Query: " + time_step["query"])
				print("True answer: " + time_step["answer"])
				predicted_answer = self.agent.answer_query(time_step["query"])
				print("Predicted answer: " + predicted_answer)
				print("Similarity: {}\n".format(cosine_similarity(get_embedding(time_step["answer"]), get_embedding(predicted_answer))))