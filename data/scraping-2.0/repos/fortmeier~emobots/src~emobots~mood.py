import openai
from .tools import get_chat_history


def mood_analysis(client, name, _chat_messages, _mood_analyis_system_prompt):
    mood_analysis_chat_history = get_chat_history(_chat_messages, name)

    mood_analysis_prompt = (
        mood_analysis_chat_history + "\n" + _mood_analyis_system_prompt
    )

    mood_analysis_messages = [{"role": "user", "content": mood_analysis_prompt}]

    mood_analysis_completion = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=mood_analysis_messages, temperature=0.3
    )
    mood_analysis_response = mood_analysis_completion.choices[0].message.content

    return mood_analysis_response
