import openai
import os
from pydantic import BaseModel, ValidationError, field_validator
from dotenv import load_dotenv

load_dotenv()

class OpenAIAPIClient:
    def __init__(self):
       
        api_key = os.getenv('OPENAI_API_KEY')
        self.client = openai.OpenAI(api_key=api_key)

    def ask_question_with_reasoning(self, question: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a validator. Explain your reasoning and provide a relevant citation."},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content

    def parse_response(self, response: str) -> dict:
        # Splitting the response into reasoning and citation
        parts = response.split(" / Citation: ")
        reasoning = parts[0].replace("Reasoning: ", "") if len(parts) > 0 else ""
        citation = parts[1] if len(parts) > 1 else ""
        return {"reasoning": reasoning, "citation": citation}

    def categorize_citation(self, citation: str, source_text: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Categorize the following citation as 'valid' or 'invalid' based on the source text."},
                {"role": "user", "content": f"Citation: {citation}\nSource Text: {source_text}"}
            ]
        )
        return response.choices[0].message.content.strip()

class AIResponse(BaseModel):
    reasoning: str
    citation: str
    source_text: str
    citation_status: str

    @field_validator('citation_status')
    @classmethod
    def validate_citation_status(cls, v: str) -> str:
        if v.lower() == 'invalid':
            raise ValidationError("Citation categorized as invalid.")
        return v

def validate_ai_response(client: OpenAIAPIClient, question: str, source_text: str):
    original_response = client.ask_question_with_reasoning(question)
    structured_response = client.parse_response(original_response)
    citation_status = client.categorize_citation(structured_response['citation'], source_text)
    structured_response['citation_status'] = citation_status

    if citation_status.lower() == 'invalid':
        print("Validation failed: Citation categorized as invalid.")
    else:
        response_model = AIResponse(**structured_response, source_text=source_text)
        print("Validation successful:", response_model)

# Example usage
client = OpenAIAPIClient()
question = "Give great qualities about India"
source_text = "donkeys are happy"
validate_ai_response(client, question, source_text)