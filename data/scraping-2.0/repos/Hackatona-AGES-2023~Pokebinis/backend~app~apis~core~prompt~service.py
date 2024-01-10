# from app.apis.core.prompt.repository import PromptRepository
from app.ext.openai import openai
import json

class PromptService:
    
	@staticmethod
	def get_topics_from_carrer(carrer: str):
		prompt = "Me de 5 topicos que eu preciso ter conhecimento avançado para desenvolver minha carreira na area: {}. De os topicos em formato JSON, com uma chave 'topics' contendo uma lista com os 5 topicos".format(carrer)
		res = openai.prompt(prompt, max_tokens=100)
		res = res.get("choices")[0].get('text').strip()
		# PromptRepository.create_carrer(carrer)
		return json.loads(res)

	@staticmethod
	def get_content_from_topic(topic: str):
		prompt = f"Escreva um texto que me faça aprender sobre {topic} para me desenvolver profissionalmente"
		text = openai.prompt(prompt, max_tokens=400)
		text = text.get("choices")[0].get('text').strip()
		print(f'Text generated for topic {topic}')
		prompt = f"Me sugira um video real da internet para aprender mais sobre o topico {topic} para meu desenvolvimento profissional. Responda somente a url do video"
		media_url = openai.prompt(prompt, max_tokens=200)
		media_url = media_url.get("choices")[0].get('text').strip()
		response = {
			'name': topic,
			'content': {
				"media_url": media_url,
				"text": text
			}
			}
		return response
		
	