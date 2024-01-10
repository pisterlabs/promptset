from app import openai


async def ask(question: str) -> str:
    answer = await openai.ask(question)
    return answer
