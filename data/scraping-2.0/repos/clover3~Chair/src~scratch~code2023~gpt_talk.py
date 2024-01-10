import openai

from utils.open_ai_api import OpenAIProxy, ENGINE_GPT_3_5, parse_chat_gpt_response, ENGINE_GPT4


def request(engine, history, new_msg):
    is_user = len(history) % 2 == 0
    messages = []
    for item in history:
        if is_user:
            e = {"role": "user", "content": item}
        else:
            e = {"role": "assistant", "content": item}
        messages.append(e)
        is_user = not is_user

    messages.append({"role": "user", "content": new_msg})
    obj = openai.ChatCompletion.create(
        model=engine, messages=messages, timeout=20,
    )
    return obj


def main():
    proxy = OpenAIProxy(ENGINE_GPT_3_5)

    last_msg = "You should not upload a paper to arxiv before the peer-review to main anonymity"
    prompt_prefix = "Counter argue this. (limit 30 words): \n"
    team_A = "A"
    team_B = "B"
    # prompt_prefix = " (40 단어 제한): \n"
    print(f"{team_A}: " + last_msg)
    for i in range(10):
        for team in [team_B, team_A]:
            prompt = prompt_prefix + last_msg
            # res = proxy.request(prompt)
            res = request(ENGINE_GPT_3_5, [], prompt)
            # print(res)
            res_text = parse_chat_gpt_response(res)
            last_msg = str(res_text)
            print(prompt)
            print(f"{team} : " + res_text)


if __name__ == "__main__":
    main()