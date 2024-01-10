```python
import openai
from src.models.language_model import GPT3Model
from src.models.sentiment_analysis import SentimentAnalysis
from src.models.named_entity_recognition import NamedEntityRecognition
from src.utils.data_encryption import encrypt_data
from src.database.mongodb_connection import MongoDBConnection

class InterviewSystem:
    def __init__(self):
        self.gpt3_model = GPT3Model()
        self.sentiment_analysis = SentimentAnalysis()
        self.named_entity_recognition = NamedEntityRecognition()
        self.db_connection = MongoDBConnection()

    def generate_interview_questions(self, context):
        prompt = f"{context}\nWhat would you like to ask next?"
        response = self.gpt3_model.generate_text(prompt)
        return response

    def analyze_sentiment(self, text):
        sentiment = self.sentiment_analysis.analyze(text)
        return sentiment

    def extract_entities(self, text):
        entities = self.named_entity_recognition.extract(text)
        return entities

    def store_interview_data(self, interview_data):
        encrypted_data = encrypt_data(interview_data)
        self.db_connection.insert_data("interviews", encrypted_data)

    def conduct_interview(self, initial_prompt):
        interview_data = {"questions": [], "answers": [], "sentiments": [], "entities": []}
        context = initial_prompt
        while True:
            question = self.generate_interview_questions(context)
            print(question)
            answer = input()
            sentiment = self.analyze_sentiment(answer)
            entities = self.extract_entities(answer)
            interview_data["questions"].append(question)
            interview_data["answers"].append(answer)
            interview_data["sentiments"].append(sentiment)
            interview_data["entities"].append(entities)
            context += f"\n{question}\n{answer}"
            if input("Continue interview? (yes/no)") == "no":
                break
        self.store_interview_data(interview_data)
        return interview_data
```