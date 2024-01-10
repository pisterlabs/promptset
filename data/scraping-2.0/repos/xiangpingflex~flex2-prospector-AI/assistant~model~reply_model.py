import openai

from assistant.api.sagemaker_client import SagemakerClient
from assistant.common.constant import (
    FINE_TUNED_GPT_35,
    FINE_TUNED_LLAMA2,
    FINE_TUNED_GPT_4,
)
from assistant.model.knowledge_base import KnowledgeBase
from assistant.model.prompts.reply_prompt import generate_reply_prompt


class ReplyLLM:
    def __init__(
        self,
        model_name: str,
        profile_name: str = None,
        region_name: str = None,
        endpoint_name: str = None,
    ) -> None:
        self.model_name = model_name
        self.profile_name = profile_name
        self.region_name = region_name
        self.endpoint_name = endpoint_name
        self.sagemaker_client = (
            SagemakerClient(
                profile_name=profile_name,
                region_name=region_name,
                endpoint_name=endpoint_name,
            )
            if model_name == FINE_TUNED_LLAMA2
            else None
        )
        self.vec_db = KnowledgeBase().create_knowledge_base()

    def retrieve_info(self, query: str):
        similar_response = self.vec_db.similarity_search(query, k=3)
        page_contents_array = [doc.page_content for doc in similar_response]
        return page_contents_array

    def generate_reply_email(
        self,
        message: str,
        max_tokens: int = 500,
        num_completions: int = 1,
        temperature: float = 0.9,
        top_p: float = 0.9,
    ) -> str:
        relevant_messages = self.retrieve_info(message)
        prompt = generate_reply_prompt(message, relevant_messages)
        if self.model_name == FINE_TUNED_GPT_35:
            print("reply generation calling gpt-3.5-turbo")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                n=num_completions,
            )
            return response["choices"][0]["message"]["content"]
        if self.model_name == FINE_TUNED_GPT_4:
            print("reply generation calling gpt-4")
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=prompt,
                max_tokens=max_tokens,
                temperature=0.9,
                n=num_completions,
            )
            return response["choices"][0]["message"]["content"]
        if self.model_name == FINE_TUNED_LLAMA2:
            print("reply generation calling llama2")
            payload = {
                "inputs": [prompt],
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "top_p": top_p,
                    "temperature": temperature,
                },
            }
            response = self.sagemaker_client.invoke_llama2_endpoint(payload)
            return response[0]["generation"]["content"]
        return ""
