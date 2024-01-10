"""
This module call openai API with user question
"""
import openai


def response_chat_gpt(user_question: str,
                      messages: list[dict[str, str]]) -> str:
    """This function call openai API with the user question
    :param user_question:
    :param messages:
    :return: str
    """

    messages.append({"role": "user", "content": user_question})

    gpt_response_question: dict = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1000,
        temperature=0.4,

    )

    text_response_assistant: str = gpt_response_question[
        "choices"][0]["message"]["content"]

    total_tokens_used: int = gpt_response_question["usage"]["total_tokens"]

    if total_tokens_used >= 3000:

        messages.clear()

        messages.insert(0, {"role": "system",
                            "content": """eres un asistente increible y
                                    eres el mas inteligente"""
                            })

        messages.insert(1, {"role": "user", "content": user_question})

    messages.append(
        {"role": "assistant", "content": text_response_assistant})

    # print(f"despues del if: {messages}")

    # print(gpt_response_question)

    return text_response_assistant
