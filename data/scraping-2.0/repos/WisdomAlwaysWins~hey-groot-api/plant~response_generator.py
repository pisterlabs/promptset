import pickle
import numpy as np
import json
import random
from langchain.tools import BaseTool

MODEL_PATH = 'static/sentence_transformer_model.pkl'
JSON_PATH = 'static/embedding_data.json'

with open(MODEL_PATH, 'rb') as f:
  model = pickle.load(f)

with open(JSON_PATH, 'r') as f:
    data = json.load(f)
    

class ResponseGenerator(BaseTool):
    name = "Response_Generator"
    description = """일상적인 대화 시 사용하는 도구"""

    def _run(self, query: str) -> str:
        return response_generator(query)
    
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("질문에 답할 수 없어요.")


def response_generator(query):
    def get_most_similar_question(query):
        query_embedding = model.encode(query)
        similarities = []

        for question in data.keys():
            question_embedding = model.encode(question)
            similarity = np.dot(query_embedding, question_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(question_embedding))
            similarities.append(similarity)

        most_similar_idx = np.argmax(similarities)
        return list(data.keys())[most_similar_idx]

    def get_response(query):
        most_similar_question = get_most_similar_question(query)
        responses = data[most_similar_question]

        # 무작위로 응답을 선택
        response = random.choice(responses)
        response_text = list(response.values())[0]

        return response_text

    return get_response(query)
