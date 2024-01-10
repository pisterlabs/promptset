from langchain.chat_models import ChatOpenAI

from app.core.config import settings
from app.generation.prompts import check_answer_template

def check_answer(question, answer):
    chat = ChatOpenAI(
            model_name="gpt-4-1106-preview",
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY,
        )
    prompt = check_answer_template.format(question=question, user_answer=answer)
    response = chat.predict(prompt)
    return response