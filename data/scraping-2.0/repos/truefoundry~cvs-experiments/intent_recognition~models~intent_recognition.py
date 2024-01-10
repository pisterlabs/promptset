import os
from openai import AzureOpenAI
import requests
import json
from dotenv import load_dotenv

from pydantic import BaseModel

from .classification_result import CategoryEnum, ClassificationResult
from .ml_classifier_utils import (
    get_classifier_model,
    get_openai_client,
    label2CategoryEnum,
)

load_dotenv()

# OPEN AI

# Constants
MODEL_DEPLOYMENT = os.getenv("MODEL_DEPLOYMENT")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")


class IntentRecognition(BaseModel):
    tfy_api_key: str

    def models(self) -> object:
        # Initialize models information
        return {
            "GPT-3-5-turbo": lambda query: self.invoke_prompt_based_model(
                model="OpenAI/gpt-3-5-turbo", query=query
            ),
            "GPT-4": lambda query: self.invoke_prompt_based_model(
                model="OpenAI/gpt-4", query=query
            ),
            "Zephyr-7b-beta": lambda query: self.invoke_finetuned_model(
                url=os.getenv("ZEPHYR_7B_FINETUNED"), query=query
            ),
            "Llama-2-7b": lambda query: self.invoke_finetuned_model(
                url=os.getenv("LLAMA_2_7B_FINETUNED"), query=query
            ),
            "Truefoundry": lambda query: self.invoke_logreg_model(query=query),
        }

    def classify(self, query: str) -> list[ClassificationResult]:
        return self.invoke_prompt_based_model(query=query)

    def classify_with_model(self, query: str, model: str) -> list[ClassificationResult]:
        return self.models()[model](query=query)

    def prompt_based_model_prompt(self, query: str) -> str:
        return f"Your job is to understand the user query, which is a simple phrase, and output which of the following three lines of business should this query be directed to. The three lines of business are: 1. Minute clinic Description: walk-in clinics (MinuteClinic), which provide basic healthcare services like vaccinations and minor illness treatment examples : Cold, Strep, Sinus Infection or Flu Symptoms, Tuberculosis (TB) Test, Ear Infection Treatment, Cold & Upper Respiratory Infection Treatment, Urinary Tract & Bladder Infection Treatment, Camp or Sports Physical, Flu Vaccine, Pink Eye Treatment, Dermatitis, Rash & Skin Irritation Treatment, Sexually Transmitted Infection (STI) Screening, Ear Wax Removal, Cough & Bronchitis Treatment, General Medical Exam, Congestion & Sinus Infection Treatment, Annual wellness exam, Sore & Strep Throat Treatment, DOT Physical, COVID-19 Treatment. 2. Drug: Description: prescription medications, over-the-counter drugs, health and wellness products, and more. 3. Product: Vitamins, Supplements, First Aid, Pain Relief, Allergy, Cold and Flu, Health Monitoring, Thermometers, Blood Pressure Monitors, Personal Care, Skin Care, Hair Care, Oral Care, Feminine Hygiene, Shaving, Grooming, Beauty, Cosmetics, Makeup, Baby Care, Diapers, Wipes, Baby Formula, Children's Medicines, Household, Cleaning, Laundry, Paper Products, Air Fresheners, Snacks, Beverages, Food, Groceries, Electronics, Batteries, Chargers, Cables, Seasonal, Holiday Items, Decorations, Gifts, Cards. which categor does {query} fall into?. return a parsable JSON. Only return the requested json. The output should be parsable directly by a json parser. Example: {json.dumps({"results": [{"category": "Product", "confidence_score": "score between 1 and 10", "explanation": "details in 5-6 words"}, {"category": "Minute Clinic", "confidence_score": "score between 1 and 10", "explanation": "details in 5-6 words"}, {"category": "Drug", "confidence_score": "score between 1 and 10", "explanation": "details in 5-6 words"}]})}"

    def invoke_prompt_based_model_with_llm_gateway(
        self, query, model="OpenAI/gpt-3-5-turbo"
    ) -> list[ClassificationResult]:
        """
        # Returns the response of the TFY LLM API depending on the llm requested
        """
        headers = {
            "Authorization": f"Bearer {self.tfy_api_key}",
        }
        body = {
            "messages": [
                {"role": "system", "content": "You are an AI bot."},
                {
                    "role": "user",
                    "content": self.prompt_based_model_prompt(query=query),
                },
            ],
            "model": {
                "name": model,
                "parameters": {
                    "temperature": 0.5,
                    "maximum_length": 1000,
                    "top_p": 1.0,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                },
            },
        }

        try:
            response = requests.post(
                os.getenv("TFY_LLM_GATEWAY"), headers=headers, json=body
            )
            data = response.json()
            text = data.get("text").strip()
            results = json.loads(text)["results"]
            return [ClassificationResult(**result) for result in results]
        except Exception:
            return [
                ClassificationResult(
                    category=CategoryEnum.NA,
                    confidence_score=0,
                    explanation="I am not sure about the category for the given input",
                )
            ]

    def finetuned_model_prompt(self, query: str) -> str:
        return f"For the below instruction and input, give the response\n\n### Instruction:\nYour job is to understand the user query, which is a simple phrase, and output which of the following three lines of business should this query be directed to. The three lines of business are: 1. Minute clinic Description: walk-in clinics (MinuteClinic), which provide basic healthcare services like vaccinations and minor illness treatment examples : Cold, Strep, Sinus Infection or Flu Symptoms, Tuberculosis (TB) Test, Ear Infection Treatment, Cold & Upper Respiratory Infection Treatment, Urinary Tract & Bladder Infection Treatment, Camp or Sports Physical, Flu Vaccine, Pink Eye Treatment, Dermatitis, Rash & Skin Irritation Treatment, Sexually Transmitted Infection (STI) Screening, Ear Wax Removal, Cough & Bronchitis Treatment, General Medical Exam, Congestion & Sinus Infection Treatment, Annual wellness exam, Sore & Strep Throat Treatment, DOT Physical, COVID-19 Treatment. 2. Drug: Description: prescription medications, over-the-counter drugs, health and wellness products, and more. 3. Product: Vitamins, Supplements, First Aid, Pain Relief, Allergy, Cold and Flu, Health Monitoring, Thermometers, Blood Pressure Monitors, Personal Care, Skin Care, Hair Care, Oral Care, Feminine Hygiene, Shaving, Grooming, Beauty, Cosmetics, Makeup, Baby Care, Diapers, Wipes, Baby Formula, Children's Medicines, Household, Cleaning, Laundry, Paper Products, Air Fresheners, Snacks, Beverages, Food, Groceries, Electronics, Batteries, Chargers, Cables, Seasonal, Holiday Items, Decorations, Gifts, Cards. which categories does the following term fall into?\n\n### Input:\n {query} \n\n### Response:"

    def invoke_finetuned_model(self, url, query) -> list[ClassificationResult]:
        try:
            # Define the URL and headers
            headers = {"accept": "application/json", "Content-Type": "application/json"}

            # Define the data payload
            data = {
                "request_id": "string",
                "prompt": self.finetuned_model_prompt(query=query),
                "stream": False,
                "n": 1,
                "best_of": 1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "repetition_penalty": 1,
                "temperature": 1,
                "top_p": 1,
                "top_k": -1,
                "min_p": 0,
                "use_beam_search": False,
                "length_penalty": 1,
                "early_stopping": False,
                "stop": ["string"],
                "stop_token_ids": [0],
                "ignore_eos": False,
                "max_tokens": 2000,
                "logprobs": 0,
                "prompt_logprobs": 0,
                "skip_special_tokens": True,
                "spaces_between_special_tokens": True,
                "return_full_text": False,
                "details": {
                    "prompt_token_ids": False,
                    "prompt_text": False,
                    "output_token_ids": False,
                    "output_text": False,
                },
            }

            # Make the POST request
            response = requests.post(url, headers=headers, json=data)

            # Check if the request was successful
            if response.status_code == 200:
                results = response.json()
                return [
                    ClassificationResult(**result)
                    for result in json.loads(results["text"][0])["results"]
                ]
        except:
            # TODO: add explicit exception handling and check for the reason of failure
            return []

    def invoke_logreg_model(self, query) -> list[ClassificationResult]:
        try:
            cl_model = get_classifier_model()
            client = get_openai_client()
            embedding = (
                client.embeddings.create(
                    input=query,
                    model="text-embedding-ada-002",
                )
                .data[0]
                .embedding
            )
            prediction_probabilities = cl_model.predict_proba([embedding])
            classification_results = []
            for i in range(len(prediction_probabilities[0])):
                classification_results.append(
                    ClassificationResult(
                        category=label2CategoryEnum[i],
                        confidence_score=prediction_probabilities[0][i],
                        explanation="Model response.",
                    )
                )
            return classification_results
        except Exception:
            return [
                ClassificationResult(
                    category=CategoryEnum.NA,
                    confidence_score=0.0,
                    explanation="Model failed to produce a response.",
                )
            ]

    def invoke_prompt_based_model(
        self, query, useLlmGateway: bool = False
    ) -> ClassificationResult:
        if useLlmGateway:
            return self.invoke_prompt_based_model_with_llm_gateway(query=query)

        return self.invoke_prompt_based_model_with_openai(query=query)

    def invoke_prompt_based_model_with_openai(self, query):
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

        response = client.completions.create(
            model=MODEL_DEPLOYMENT,
            prompt=self.prompt_based_model_prompt(query=query),
            max_tokens=1000,
        )
        data = response.choices[0].text

        return ClassificationResult(json.loads(data))
