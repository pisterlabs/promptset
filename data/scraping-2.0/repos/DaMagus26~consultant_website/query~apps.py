from django.apps import AppConfig
from legal_answer_extraction.answer_extractor import AnswerExtractor
from legal_answer_extraction.qa_model.openai_qa import OpenAIModel
from legal_answer_extraction.vector_db import weaviate_db


model = None


class QueryConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "query"

    def ready(self):
        global model
        finder = OpenAIModel(base_url='https://api.proxyapi.ru/openai/v1', api_key='sk-uJPlEJyqRg8jeYD1rSRxzV0DyXHzQtNb')
        model = AnswerExtractor(finder)

