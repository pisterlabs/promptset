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


seeds = get_datas('data/seed_data_v2.jsonl')
# augmented_dials = get_datas('./augmented_dial_gpt-4.jsonl')
augmented_dials = []

answers = []
for iter in tqdm(range(20), desc=f"Completing..."):
    rand_seeds_indies = []
    seeds_pool = []
    while len(rand_seeds_indies) < 6:
        index = random.randint(0, len(seeds) - 1)
        if index not in rand_seeds_indies:
            rand_seeds_indies.append(index)
            seeds_pool.append(seeds[index])
    rand_augs_indies = []
    augs_pool = []
    # while len(rand_augs_indies) < 2:
    #     index = random.randint(0, len(augmented_dials) - 1)
    #     if index not in rand_augs_indies:
    #         rand_augs_indies.append(index)
    #         augs_pool.append(augmented_dials[index])

    samples_pool = seeds_pool
    random.shuffle(samples_pool)
    # for i in range(len(seeds)):
    #     data = seeds[i]
    #     target_dialogue = data['dialogue']
    #     try:
    #         bot_index = target_dialogue.rindex("bot:")
    #         data['dialogue'] = target_dialogue[:bot_index].strip()
    #     except:
    #         pass

    # 본인 프롬프트에 맞게 수정
    prompt = f'''### Instruction: Please generate a new dialogue that 'user' is asking for recommendations 'bot'. 'bot' should respond like a dialogue agent that requests more information for a better recommendation rather than recommend directly. You should follow the structure of given samples; You should avoid creating similar bot responses in given samples, Please diversely create bot responses that request new information to users. \n ### Input: [Sample 1] {samples_pool[0]['dialogue']} \n [Sample 2] {samples_pool[1]['dialogue']} \n [Sample 3] {samples_pool[2]['dialogue']} \n [Sample 4] {samples_pool[3]['dialogue']} \n [Sample 5] {samples_pool[4]['dialogue']} \n [Sample 6] {samples_pool[5]['dialogue']} \n ### Output: [Sample 7] '''

    answer, usage = run_gpt_turbo(ENGINE, prompt=prompt)
    answers.append({'prev_state': "", "dialogue": answer, "cur_state": "", "response": ""})
    augmented_dials.append({'prev_state': "", "dialogue": answer, "cur_state": "", "response": ""})

with open('data/augmented_dial_v2_6shots_temp1.0_pp0.3_gpt-4.jsonl', 'w', encoding='utf-8') as fw:
    for data in answers:
        fw.write(json.dumps(data, ensure_ascii=False))
        fw.write("\n")