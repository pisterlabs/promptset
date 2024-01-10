from dataclasses import asdict

from news_powered_by_metaphor.config import METAPHOR_API_KEY
from metaphor_python import Metaphor

from news_powered_by_metaphor.services.openAI.service import OpenAIService



class MetaphorService:
    def __init__(self, use_prompt=0):
        self.client = Metaphor(api_key=METAPHOR_API_KEY)
        self.openai_service = OpenAIService()

    def request_search(self, input_query, include_domains, *args, **kwargs):
        response = self.client.search(input_query, include_domains=include_domains, num_results=5, *args, **kwargs)
        content_result = response.get_contents()

        print("==========metaphor search response:==========\n", response)
        response_dict = asdict(response)
        for index, content in enumerate(content_result.contents):
            response_dict["results"][index]["summary"] = self.openai_service.request_description_from_content(content.extract)

        return response_dict["results"]