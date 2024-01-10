import google.generativeai as genai
from django.conf import settings
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI


class Chat:
    @classmethod
    def gemini(cls, prompt: str, model: str, assistant: list = None):
        google_api_key = settings.GOOGLE_API_KEY
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel(model)
        messages = [prompt]
        if assistant is not None and isinstance(assistant, list):
            messages = assistant + messages
        response = model.generate_content(messages)
        return response.text

    @classmethod
    def openai(cls, prompt: str, model: str, assistant: list = None):
        messages = [{"role": "user", "content": prompt}]
        if assistant is not None and isinstance(assistant, list):
            messages = assistant + messages
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        chat_completion = client.chat.completions.create(model=model, messages=messages)
        return chat_completion.choices[0].message.content


class LangChain:
    @classmethod
    def translate(cls, sentence: str, target_language: str = "English"):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a translator bot. Translate sentences to {target_language}.",
                ),
                (
                    "human",
                    "Translate this: {sentence}",
                ),
            ]
        )
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro", convert_system_message_to_human=True
        )
        chain = prompt | llm
        response = chain.invoke(
            {
                "target_language": target_language,
                "sentence": sentence,
            }
        )
        return response.content
