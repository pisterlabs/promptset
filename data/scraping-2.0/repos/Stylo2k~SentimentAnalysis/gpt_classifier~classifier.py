from classes import Classifier
from starlette.requests import Request


from ray import serve
from fastapi import FastAPI
import os
import openai

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

openai.organization = os.getenv("OPENAI_API_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")


app = FastAPI()


@serve.deployment
class GptClassifier(Classifier):
    def __init__(self):
        self.model = "text-davinci-003"

    def classify(self, text : str):
        
        response = openai.Completion.create(
            model = f"{self.model}",
            prompt= f"Use JSON to format the response like this:\n\n{{\"sentiment\": \"sentiment here\", reason: \"reason here\"}}. Classify the sentiment of the following sentence and given me the reason:\n\n{text}\n",
            temperature=0,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        return response['choices'][0]['text']

    async def __call__(self, http_request: Request) -> str:
        text: str = await http_request.json()
        return self.classify(text)


class GptPreProcessor:
    def __init__(self):
        pass

    def preprocess(self, text : str):
        return text

@serve.deployment
@serve.ingress(app)
class GptDeployment:
    def __init__(self, preprocessor, classifier):
        self.preprocessor = preprocessor
        self.classifier = classifier

    async def classify(self, text : str):
        preprocessed_text = self.preprocessor.preprocess(text)
        ref = await self.classifier.classify.remote(preprocessed_text)
        return await ref

    @app.post("/gpt")
    async def call(self, http_request: Request):
        text: str = await http_request.json()
        return self.classify(text)

gpt = GptDeployment.bind(
    GptPreProcessor(), 
    GptClassifier.bind()
)