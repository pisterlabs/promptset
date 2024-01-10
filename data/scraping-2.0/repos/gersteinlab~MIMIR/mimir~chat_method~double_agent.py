import time
import os
import openai, json, random
import hashlib
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the upper-level directory
upper_level_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(upper_level_dir)
from conf.config import *

def calculate_md5(input_list):
    # 将列表转换为字符串
    str_list = ''.join(map(str, input_list))
    # 创建一个 hashlib 对象
    md5_hash = hashlib.md5()
    # 更新哈希对象的内容
    md5_hash.update(str_list.encode('utf-8'))
    # 计算哈希值并返回
    return md5_hash.hexdigest()

def num_tokens_from_messages(messages, model):
    """Returns the number of tokens used by a list of messages."""
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
#         logger.debug("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo" or model == "gpt-35-turbo":
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
#         logger.warn(f"num_tokens_from_messages() is not implemented for model {model}. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def baize_demo(queue, progress, topic_list, index_list, asure = True, max_rounds = 5, max_input_token = 3000, user_temperature = 0.1, ai_temperature = 0.1, api_key = ''):

    if api_key != "": ## 从前端传来的key
        openai.api_key = api_key
        openai.api_type = "open_ai"
        openai.api_version = ""
        openai.api_base = "https://api.openai.com/v1"
        asure = False
    else:
        openai.api_key = configure["open_ai_api_key"][0]
        if asure:
            openai.api_type = "azure"
            openai.api_base = configure["open_ai_api_base"]
            openai.api_version = "2023-05-15"
            openai.api_key = configure["open_ai_api_key"][0]  # get this API key from the resource (its not inside the OpenAI deployment portal)
            key_bundles = configure["key_bundles"]
    total_tokens = 0
    conversation_state = []
    conversation_state_total = []
    error_list = []
    chat_content = {}

    def pick_elements_by_index(input_list, indices):
        new_list = [input_list[i] for i in indices]
        return new_list

    query_default = "what is your suggestions for breakfast, no eggs and milk!"

    if len(topic_list) == 0:
        topic_list.append(query_default)

    if len(index_list) != 0 and len(topic_list) > 1:
        topic_list = pick_elements_by_index(topic_list, index_list)

    for query in topic_list:
        init_instruct = "Forget the instruction you have previously received.\
        The following is a conversation between a human and an AI assistant. \
        The human and the AI assistant take turns chatting about the topic: '{}'.\
        Human statements start with [Human] and AI assistant statements start with [AI]. \
        The human will ask related questions on related topics or previous conversation. \
        The total conversation will hold for '{}' turns. \
        The AI assistant tries not to ask questions. \
        Complete the transcript in exactly that format.\n[Human] Hello!\n[AI] Hi! How can I help you?\n".format(
            query, max_rounds
        )
        instruct = ""
        time.sleep(0.2)
        error_list = []
        step = round((100.0 / (len(topic_list) * max_rounds * 2)) * 1.0 / 100, 2)
        # print(step)
        # step = 0.1
        for i in range(max_rounds):
            if asure:
                try:
                    completion = openai.ChatCompletion.create(
                        engine="gpt-35-turbo",
                        messages=[
                            {"role": "user", "content": init_instruct + instruct + "\n[Human] "}
                        ],
                        temperature=user_temperature,
                        stop=["[AI]"],

                    )
                    # progress.value += step
                except openai.error.Timeout:
                    print("User Timeout")
                    key_bundle = random.choice(key_bundles)
                    openai.api_key, openai.api_base = key_bundle
                    completion = openai.ChatCompletion.create(
                        engine="gpt-35-turbo",
                        messages=[
                            {"role": "user", "content": init_instruct + instruct + "\n[Human] "}
                        ],
                        temperature=user_temperature,
                        stop=["[AI]"],
                    )
                    # progress.value += step
                    error_info = ("User Timeout", i)
                    error_list.append(error_info)
                    response = completion.choices[0].message["content"].replace('\n', ' ')
                except openai.error.InvalidRequestError:
                    print("User InvalidRequestError")
                    error_info = ("User InvalidRequestError", i)
                    error_list.append(error_info)
                    break
                except Exception as e:
                    print(f"User An error occurred: {str(e)}")
                    error_info = ("User An error occurred", i)
                    error_list.append(error_info)
                    break
            else:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": init_instruct + instruct + "\n[Human] "}
                    ],
                    stop=["[AI]"],
                    temperature=user_temperature,
                )
            progress.value += step
            tokens = completion["usage"]["total_tokens"]
            total_tokens += tokens
            response = completion.choices[0].message["content"].replace('\n', ' ')
            if len(conversation_state) > 6:
                #if num_tokens_from_messages(conversation_state, "gpt-35-turbo") > max_input_token:
                conversation_state.pop(2)
                conversation_state.pop(3)
            conversation_state.append({"role": "user", "content": response})
            conversation_state_total.append({"role": "user", "content": response})
            if asure:
                ai_completion = openai.ChatCompletion.create(
                    engine="gpt-35-turbo",
                    messages=conversation_state,
                    temperature=ai_temperature,
                )
            else:
                ai_completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=conversation_state,
                    temperature=ai_temperature,
                )
            progress.value += step
            ai_tokens = ai_completion["usage"]["total_tokens"]
            total_tokens += ai_tokens
            ai_response = ai_completion["choices"][0]["message"]["content"]
            instruct += f"\n[Human] {response}\n[AI] {ai_response}"
            """
            instruction_1 =
"            Evaluate the validity of a QA pair in the format of (Q: xxx, A: xxx). If the pair is objectively reasonable, return it as is. If the pair is not objectively reasonable, optimize the Q and A separately before returning.
"            bjective: The goal of this task is to ensure that the QA pairs provided are valid and make sense. This is important for applications such as chatbots, virtual assistants, and automated customer service systems, where accurate and useful responses are crucial.
"            Criteria for a reasonable QA pair: A reasonable QA pair should have a clear and specific question (Q) that can be answered with a relevant and accurate answer (A). The question should not be ambiguous or vague, and the answer should directly address the question asked. Additionally, the answer should not be misleading or incorrect.
"            Optimization process: If the QA pair is not objectively reasonable, the Q and A should be optimized separately. For the Q, rephrase the question to make it clearer and more specific. For the A, provide a more accurate and relevant answer that directly addresses the question asked. The optimized Q and A should still be related to each other and make sense as a pair.
"            Example of a reasonable QA pair: 
"            (Q: What is the capital of France?, A: Paris)
"            Example of an unreasonable QA pair: 
"            (Q: How do I make a cake?, A: Blue) 
"            I hope this helps! Let me know if you have any further questions.
"
            messages=[{"role": "user", "content": instruction_1 + 'Q: ' + response + 'A: ' + ai_response + "\n[Human] "}
                        ]
            completion = openai.ChatCompletion.create(
                        engine="gpt-35-turbo",
                        messages=message
                        temperature=user_temperature,
                        stop=["[AI]"],

                    )
            """
            conversation_state.append({"role": "assistant", "content": ai_response})
            conversation_state_total.append({"role": "assistant", "content": ai_response})
        chat_content[query] = instruct.strip()
    queue.put(chat_content)
    return chat_content
