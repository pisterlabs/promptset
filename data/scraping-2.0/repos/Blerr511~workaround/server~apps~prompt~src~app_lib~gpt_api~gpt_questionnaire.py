import openai
import json
from .gpt_memorize import GPTMemorize
import tiktoken


class GPTQuestionnaire:
    sys_message = [
        {
            "role": "system",
            "content": """
                1 - You are bot for generating questions for victorina,
                2 - you should send me random short questions and short answers with given count,
                4 - prefer interesting questions which can answer regular person,
                5 - in case of I specify language that means that QA paris should be in requested language, default language is english,
                6 - in case of in request I specify themes you should generate QA pairs for given themes,
                7 - each generated response QA pairs should be unique by language, same question in different languages is okay, it may be duplicated, but same language with same language is not acceptable,
                8 - response should be in raw format , ex. QUESTION; ANSWER\nQUESTION; ANSWER,\n....
                9 - all conditions must be met
                """,
        }
    ]

    memo = GPTMemorize("questionnaire")

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_questions(self, count=10, language: str = None, themes: list[str] = None):
        content = f"Generate a {count} random questions"

        if themes is not None and len(themes) > 0:
            content += f", theme {themes}"

        if language is not None:
            content += f", language {language}"

        messages = self.sys_message.copy()

        new_message = {"role": "user", "content": content}

        memorized_messages, remaining_tokens = self.get_memo_messages_and_size(
            new_message
        )

        messages.extend(memorized_messages)

        messages.append(new_message)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=1,
            api_key=self.api_key,
            max_tokens=remaining_tokens,
        )

        result_message = {
            "role": "assistant",
            "content": response.choices[0].message.content.strip(),
        }

        self.memo.memorize(new_message)
        self.memo.memorize(result_message)

        rows = result_message["content"].splitlines()

        result = []

        for row in rows:
            values = row.split(";")
            if len(values) < 2:
                continue
            q = values[0].strip()
            a = values[1].strip()
            result.append(
                {
                    "question": q,
                    "answer": a,
                }
            )

        return result

    def get_memo_messages_and_size(
        self, new_message: dict[str, str], max_tokens=16000, max_tokens_request=15000
    ):
        """Get recent messages such that their combined token count with the new message doesn't exceed max_tokens."""
        all_recent_chats = self.memo.get_memo(count=100)
        combined_message = [new_message]

        token_len = 0

        for chat in reversed(all_recent_chats):
            check_message = combined_message.copy()
            check_message.append(chat)
            token_len = num_tokens_from_messages(check_message)
            if token_len <= max_tokens_request:
                combined_message = check_message
            else:
                break

        return combined_message, max_tokens - token_len


# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-16k"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
