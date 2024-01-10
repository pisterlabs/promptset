import openai
import os
import json
import tiktoken


def count_token(text: str):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))


def count_tokens(texts: list):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return sum(len(encoding.encode(text['content'])) for text in texts)


def get_response_openai(prompt: str,
                        document: str):
    prompt_tokens, generated_tokens = 0, 0
    SYSTEM1 = prompt
    ASSISTANT1 = document
    USER1 = "Generate five different scenarios you recommending which service might fit the counselee based on the situation."
    prompt_tokens += count_token(SYSTEM1)
    prompt_tokens += count_token(ASSISTANT1)
    prompt_tokens += count_token(USER1)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": SYSTEM1},
                {"role": "assistant",
                 "content": ASSISTANT1},
                {"role": "user",
                 "content": USER1
                 }
            ],
            presence_penalty=0.6
        )
    except:
        print("OpenAI Response Error")
        return None, None, None

    generated_tokens += count_token(response['choices'][0]['message']['content'])

    SYSTEM2 = """
            You are a translator from English to Korean.Translated tone must be conversational.
            Do not translate the answers.
            Just translate the situations of counsellee only.
            You generate all text in Korean.
            Drop out the answers from counselor.
            You generate all text in Korean.
            Translation only include the situations of counsellee nothing else.
            """
    ASSISTANT2 = f"{response['choices'][0]['message']['content']}"
    USER2 = """
            Drop out the answer from counselor.
            Extract only the situations from the Counsellee and translate them into Korean.
            Tone must be conversational.
            Number them 1, 2, 3, 4, 5.
            """

    prompt_tokens += count_token(SYSTEM2)
    prompt_tokens += count_token(ASSISTANT2)
    prompt_tokens += count_token(USER2)

    try:
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                    "content": SYSTEM2},
                {"role": "assistant",
                    "content": ASSISTANT2},
                {"role": "user",
                 "content": USER2
                 }
            ],
        )
    except:
        return None, None, None

    generated_tokens += count_token(second_response['choices'][0]['message']['content'])

    return second_response['choices'][0]['message']['content'], prompt_tokens, generated_tokens


def get_chat_response(doc_dir: str):
    PROMPT = """
        You are a counseler recommending the given service for who is asking for which welfare service might fit him.
        Counselee is ignorant to the given service, you are recommending the service for the first time.
        Situation explained by counsellee MUST NOT include the name of the service.
        Situation explained by counsellee include just the situation of the counselee, MUST NOT include the content of the service.
        Situation explained by counsellee is in conversational tone.
        Situation explained by counsellee ais in very easy word.
        Situation explained by counsellee is very short.
        Situation explained by counsellee is very general, never specific.
        Counsellee's job, gender, age, disability and other attributes are various.
        Question by counsellee is related to the situation of the counsellee, NOT about the service.
        Answer is extremely short, less than 5 words.
    """

    with open(doc_dir, 'r', encoding='utf-8') as f:
        document = f.read()

    questions_generated = get_response_openai(PROMPT, document)
    return questions_generated


if __name__ == '__main__':
    with open("server/config.json") as config_file:
        config_data = json.load(config_file)
        openai.api_key = config_data["chatgpt"]["secret"]

    with open("data/embeds/embeddings_notag.json") as file:
        embeddings = json.load(file)

    price = 0
    input_dir = 'data/notags'
    data = {}
    for i, doc_dir in enumerate(sorted(os.listdir('data/notags'))):
        if i < 50:
            continue
        file_dir = os.path.join(input_dir, doc_dir)
        data_tokens = embeddings[str(i)]['tokens']
        response, p_tok, g_tok = get_chat_response(file_dir)
        try:
            price += (p_tok * 0.0015 + g_tok * 0.002) / 1000
        except:
            continue
        data[i] = {
            "filename": embeddings[str(i)]['filename'],
            "title": embeddings[str(i)]['title'],
            "index": i,
            "response": response,
        }
        print(response)
        print("Price: ", price, "Dollars ", price * 1267.14, "Won")
        print("Progress: " + str(i / len(os.listdir('data/notags')) * 100) + "%")
        with open(f'data/augmented/generated.json', 'w') as f:
            json.dump(data, f, indent=2)
        print("\n\n\n")
