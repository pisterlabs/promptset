from abc import ABC, abstractmethod
import openai
from ctransformers import AutoModelForCausalLM

from API_KEYS import CHATGPT_API_KEY


class BaseLLM(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def inference(self, input_text):
        pass

    # Add other methods here


class GPT_LLM(BaseLLM):
    def __init__(self):
        openai.api_key = CHATGPT_API_KEY

    def inference(self, input_text):
        # Process the text with LLM and return the response
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a conversational assistant. You should asnwer every question like a real human, like a conversation. Keep the answers short.",
                },
                {"role": "user", "content": input_text},
            ],
        )
        answer = completion.choices[0].message
        print(answer["content"])
        return answer["content"]


class LLaMA(BaseLLM):
    def __init__(self):
        self.llm = AutoModelForCausalLM.from_pretrained(
            "./",
            model_file="llama-2-7b-chat.Q5_K_S.gguf",
            model_type="llama",
            gpu_layers=50,
        )

    def inference(self, input_text):
        # prompt = f"You are a highly knowledgeable assistant trained to communicate effectively with humans. When answering questions, your responses should be concise and informative, mimicking a natural human conversation. The question might have typo's, might be complete or broken but you know that there is always a question. First try to guess the correct question, then answer it. Here's the first question: {input_text}"
        prompt = f"You are a conversational assistant. Asnwer every question like a real human, like a conversation. Keep the answers in two lines.\n Tell me {input_text}"
        answer = self.llm(prompt)
        return answer

    # Implement any other LLaMA-specific methods here
