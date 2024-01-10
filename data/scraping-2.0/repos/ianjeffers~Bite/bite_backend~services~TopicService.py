import json
from services.OpenAIService import OpenAIService

class TopicService:
    def __init__(self):
        self.openai_service = OpenAIService()

    def generate_related_topics(self, topic, reason):
        
        # dummy_related_topics = ["Maths", "Stats", "Geo"]
        # return dummy_related_topics

        if reason == 'mastered':
            prompt = f'Given that a user has mastered the topic "{topic}", generate a list of related topics under 10 characters they might be interested in studying next in the following format: {{"related_topics": ["Topic1", "Topic2", "Topic3"]}}.'
        elif reason == 'interest':
            prompt = f'Given that a user has expressed interest in the topic "{topic}", suggest some related topics under 10 characters that they might find interesting in the following format: {{"related_topics": ["Topic1", "Topic2", "Topic3"]}}.'
        else:
            prompt = f'Please suggest a list of diverse topics under 10 characters for a user who is open to studying anything in the following format: {{"related_topics": ["Topic1", "Topic2", "Topic3"]}}.'

        gpt_response = self.openai_service.generate_json(prompt, 250)
        if gpt_response is None:
            return {'message': 'Could not generate content'}, 500
        related_topics = json.loads(gpt_response.choices[0].text)['related_topics']
        
        return related_topics
