import json
import openai
from utils import post_msg_llm, url2, remove_not_in, extract_list, remove_in
from prompts import get_choosing_prompt, get_choosing_prompt_zero_shot, get_words_score_prompt


OPEN_API_KEY = "FAKE_KEY"
# OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
# openai.api_base = "http://10.58.0.2:6677/v1"
# openai.api_key = OPEN_API_KEY

headers = {
    "Content-Type": "application/json",
    # "Authorization": f"{OPEN_API_KEY}"  # 替换 YOUR_OPEN_API_KEY 为你的实际 API key
}


def simple_choosing_task(input_text, input_list):
    choosing_prompt = get_choosing_prompt(input_text, input_list)
    print(choosing_prompt)
    get_response = post_msg_llm(choosing_prompt, url2)
    get_list = json.loads(extract_list(get_response))
    return get_list, get_response


def score_choosing_task(input_text, input_list):
    choosing_prompt = get_words_score_prompt(input_text, input_list)
    print(choosing_prompt)
    get_response = post_msg_llm(choosing_prompt, url2, temperature=0.4)
    return get_response


def modified_choosing_task(input_text, input_list, epochs=3):
    print(input_list)
    last_len = -2
    last_list = input_list
    print(f"inputs:{input_list}")
    word_list = []
    # print(f"input:{input_list}")
    for i in range(epochs):
        word_list, raw = simple_choosing_task(input_text, last_list)
        if len(word_list) == last_len:
            break
        last_list = word_list
        last_len = len(word_list)
        # print(last_list_str)
    return remove_not_in(word_list, input_list), remove_in(word_list, input_list)

