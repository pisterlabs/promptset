import openai as openai

from SmartFriend.helpers import MAKE_SUMMARY, OPENING_TEXT, BONUS_CONDITION, FALLBACK_TEXT


def fallback_for_ai(response, conversation):
    if "ai" in response.lower():
        conversation.append(
            {
                "role": "system",
                "content": FALLBACK_TEXT,
            }
        )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation,
        )

        return response.choices[0].message.content

    return response


def get_response(conversation):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
    )

    response = fallback_for_ai(
        response.choices[0].message.content,
        conversation
    )

    return response


def start_conversation(summary):
    conversation = [
        {
            'role': 'system',
            'content': BONUS_CONDITION + " " + summary,
        },
        {
            'role': 'system',
            'content': OPENING_TEXT,
        },
    ]

    return conversation


def post_summary(conversation):
    if conversation:
        result = {
            'role': 'system',
            'content': MAKE_SUMMARY
        }

        return result

    return ""
