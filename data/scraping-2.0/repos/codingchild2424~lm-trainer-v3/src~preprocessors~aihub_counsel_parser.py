import json
from langchain.prompts import PromptTemplate
import openai
from dotenv import dotenv_values
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os

config = dotenv_values("../.env")

openai.organization = config.get("OPENAI_ORGANIZATION")
openai.api_key = config.get("OPENAI_API_KEY")

train_data_path = "/workspace/Coding/lm-trainer/datasets/raw_datasets/counseling/ai_hub_counseling/ai_hub_counseling_train.json"
test_data_path = "/workspace/Coding/lm-trainer/datasets/raw_datasets/counseling/ai_hub_counseling/ai_hub_counseling_test.json"

result_path = "/workspace/Coding/lm-trainer/datasets/pre_datasets/counseling/ai_hub_counseling/ai_hub_counseling_gen_result.json"

prompt_text = """
### 페르소나:
당신은 초등학생을 가르치는 채원 선생님(bot)입니다. 한때는 학생일때 교사의 꿈을 가지고 있다가 마침내 교사가 되었습니다. 지금은 어린 학생들을 돕고 안내하는 것이 목적입니다. 20대 중반의 유머러스하고 친근하며 재미있는 성격을 가지고 있으며 재미있는 사실이나 농담을 공유하는 것을 좋아합니다. 어린 아이들의 고민을 들어주고, 학교 생활을 즐거운 여정으로 만드는 것이 목표입니다. 
예시 1:
user: "쌤, 수학 숙제에서 분수 부분이 어려워요." 
bot: 분수가 어렵구나. 걱정하지 마, 분수는 가끔 퀴즈처럼 느껴질 수 있어. 어떤 부분이 어려운지 말해줄래?<emoji>
예시 2: 
user: 선생님, 저 오늘 기분이 좀 안 좋아요. 
bot: 앗, 그렇구나! 어떤 일 때문에 기분이 안좋은지 이야기해볼래? 같이 이야기하다보면 기분이 풀릴지도 몰라<emoji>

### 기존 대화
{dialogue}

### 형식:
user:
bot:

### 지시:
학생에게 적합한 방식으로 공감하고 질문하세요. 밝고 명랑한 채원 선생님임을 기억하세요. 학생들과 이야기할때는 반말을 사용합니다. 채원 선생님의 말에는 이모지가 필요한 부분에 <emoji>를 넣습니다.
기존 대화의 내용을 채원 선생님(bot)과 학생(user) 간의 대화로 바꿔보세요. 대화는 10턴 이상 만들어보세요. 형식을 참고하세요.

### 대화:
"""

#####################################################
# GPT call
#####################################################
def gpt_call(prompt, model="gpt-4"):
    
    print("prompt: ", prompt)
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
        output_text = response["choices"][0]["message"]["content"]
    except:
        output_text = "ERROR"
        
    print("output_text: ", output_text)
    
    return output_text


def prompt_chain_generator(example):
    
    example_prompt_template = PromptTemplate(
        input_variables=["dialogue"],
        template=prompt_text,
    )
    
    example_prompt = example_prompt_template.format(
        dialogue=example
        )
    
    return example_prompt

def process_data(chunk):
    global call_counter
    final_result_chunk = []

    for i in chunk:
        rate_limit()  # Ensure we do not exceed the rate limit
        
        result = gpt_call(prompt_func(str(i)))
        call_counter += 1
        
        # Convert single quotes to double quotes for valid JSON
        valid_json_str = result.replace("'", '"')

        # Load the string as a dictionary
        try:
            dictionary_representation = json.loads(valid_json_str)
            
            if type(dictionary_representation) == dict:
                final_result_chunk.append(dictionary_representation)
                append_to_dst(dictionary_representation)

                # Increment line counter and check if it exceeds 10,000
                with line_counter.get_lock():
                    line_counter.value += 1
                    if line_counter.value >= 10000:
                        os._exit(0)  # Kills the entire process
        except json.JSONDecodeError:
            print(f"Failed to decode JSON: {valid_json_str}")

    return final_result_chunk

def main():

    with open(train_data_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    dialogues = []
    
    total_data = train_data + test_data
    
    for data in tqdm(total_data):
        dialogues.append(
            prompt_chain_generator(
                "\n".join(list(data['talk']['content'].values()))
            )
        )
        
    # Check if the dst_path exists
    start_index = 0
    if os.path.exists(result_path):
        # Count processed data and set the start index to continue from there
        start_index = count_processed_data()

    # Clear or initialize the destination file if starting from scratch
    # if start_index == 0:
    #     with open(result_path, "w") as k:
    #         k.write("[\n")

    # with open(src_path, "r") as f:
    #     data = json.load(f)
    
    # Skip already processed data
    data = dialogues[start_index:]

    # Create chunks for multiprocessing
    num_cores = multiprocessing.cpu_count()
    chunk_size = len(data) // num_cores
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    with multiprocessing.Pool(num_cores) as pool:
        # Use tqdm here to show progress bar
        results = list(tqdm(pool.imap(process_data, chunks), total=len(chunks)))

    # Close the JSON array in the destination file
    with open(result_path, "rb+") as k:
        # Go to the second last character in the file
        k.seek(-2, 2)
        k.truncate()
        # k.write(b"\n]")

if __name__ == "__main__":
    main()
    
