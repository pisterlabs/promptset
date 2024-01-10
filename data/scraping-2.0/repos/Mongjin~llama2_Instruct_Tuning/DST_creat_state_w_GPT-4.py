from openai import OpenAI
import json
import random
from tqdm import tqdm


def get_gpt4_api_key(key_path):
    with open(key_path, 'r', encoding='utf-8') as fr:
        return fr.readline().strip()


API_KEY = get_gpt4_api_key('./GPT-4_API_Key.txt')  ## 본인 API key 입력
ENGINE = 'gpt-4-1106-preview'  ## 지금 버전이 gpt-4 turbo!
client = OpenAI(api_key=API_KEY)


def run_gpt_turbo(engine, prompt):
    completion = client.chat.completions.create(
        model=engine,
        messages=[{"role": "user", "content": prompt}],  # 입력 prompt
        max_tokens=2048,
        temperature=1.0,
        top_p=1,
        n=1,
        presence_penalty=0.3,
        frequency_penalty=0,
        # logit_bias=
        # stop=[],
    )

    print(completion)

    answer = completion.choices[0].message.content
    usage = completion.usage

    return answer, usage


def run_text_davinci(engine, prompt, max_tokens, temperature, top_p,
                     frequency_penalty, presence_penalty, logprobs, n, best_of, stop_sequences=None, debug=False
                     ):
    response = None

    try:
        prompt += "\n"
        response = client.completions.create(
            engine=engine,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop_sequences,
            logprobs=logprobs,
            n=n,
            best_of=best_of)
        if debug:
            return response["choices"][0]["text"], response["usage"]["total_tokens"], response
        return response["choices"][0]["text"], response["usage"]["total_tokens"]

    except Exception as e:
        print(e)
        return None, None


def get_datas(file_path):
    datas = []
    with open(file_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            datas.append(json.loads(line))
    return datas


seeds = get_datas('data/seed_data_state_v2.jsonl')
# augmented_dials = get_datas('./augmented_dial_gpt-4.jsonl')
augmented_dials = []

answers = []
# for iter in tqdm(range(20), desc=f"Completing..."):
    # rand_seeds_indies = []
    # seeds_pool = []
    # while len(rand_seeds_indies) < 6:
    #     index = random.randint(0, len(seeds) - 1)
    #     if index not in rand_seeds_indies:
    #         rand_seeds_indies.append(index)
    #         seeds_pool.append(seeds[index])
    # rand_augs_indies = []
    # augs_pool = []
    # while len(rand_augs_indies) < 2:
    #     index = random.randint(0, len(augmented_dials) - 1)
    #     if index not in rand_augs_indies:
    #         rand_augs_indies.append(index)
    #         augs_pool.append(augmented_dials[index])

samples_pool = seeds
random.shuffle(samples_pool)
# for i in range(len(seeds)):
#     data = seeds[i]
#     target_dialogue = data['dialogue']
#     try:
#         bot_index = target_dialogue.rindex("bot:")
#         data['dialogue'] = target_dialogue[:bot_index].strip()
#     except:
#         pass

for sample in tqdm(samples_pool, desc=f"Completing..."):
    dialogue = sample['dialogue']
    splited_dialogue = dialogue.split("user: ")
    restore_dial = ""
    prev_state = "None"
    current_state = ""
    for dial in splited_dialogue[1:]:
        restore_dial += "user: " + dial
        # 본인 프롬프트에 맞게 수정
        prompt = f'''### Instruction: Please creat a current dialogue state base on the last bot response. Please be mindful our dialogues have chariteristic that user ask a food or travel recommendation bot. Previous State denotes the dialogue state that is based on except that last utterance of user and bot. You should create a dialogue state that helps to request additional information to user for better recommendation. You should create a state following key-value format; Please diversely create a dialogue state so that bot can request new information to users. You should update Current State from Previous State and creat new states to request additional information to user. \n ### Input: [Previous State] 'prev_state': {prev_state} [Dialogue] {restore_dial} \n ### Output: [Current State] 'current_state': '''
        answer, usage = run_gpt_turbo(ENGINE, prompt=prompt)
        answers.append({'prev_state': prev_state, "dialogue": restore_dial, "cur_state": answer, "response": ""})
        augmented_dials.append({'prev_state': "", "dialogue": answer, "cur_state": "", "response": ""})
        prev_state = answer

with open('data/seed_data_state_v2_temp1.0_pp0.3.jsonl', 'w', encoding='utf-8') as fw:
    for data in answers:
        fw.write(json.dumps(data, ensure_ascii=False))
        fw.write("\n")