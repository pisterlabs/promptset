from openai import OpenAI
from dotenv import load_dotenv
from api import generate_transcript_by_recipe_name
import csv
import tiktoken

load_dotenv()

def _generate_recipe_by_recipe_name(recipe_name):
    generate_transcript_by_recipe_name(recipe_name)

def _get_top_3_video_ids(recipe_name):
    _generate_recipe_by_recipe_name(recipe_name)
    video_ids = []

    # CSV 파일에서 헤더 id에 있는 ID 가져오기
    with open('data/'+recipe_name+'/id.csv', 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_id = row.get('id')
            if video_id:
                video_ids.append(video_id)
                if len(video_ids) >= 3:
                    break

    return video_ids

def _generate_recipe_summary(recipe_name, video_id, max_context_length=3400):
    # Read the contents of the text file
    with open(f'data/{recipe_name}/{video_id}.txt', 'r') as file:
        recipe = file.read()

    # Calculate the token count
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = len(tokenizer.encode(recipe))

    # Check if the token count exceeds the maximum context length
    if tokens > max_context_length:
        print(f"Skipping generation for {recipe_name}/{video_id}.txt due to exceeding maximum context length.")
        return None

    client = OpenAI()

    # Send the command to GPT
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a recipe assistant. "},
            {"role": "user", "content": f"다음 레시피에서 말하는 필요한 재료와 양을 표기해줘, 그리고 각 단계별로 요약정리해서 말해줘:{recipe}\n"}
        ],
        max_tokens=600
    )
    return completion.choices[0].message.content


def get_recipe_summarys_from_data(recipe_name):
    video_ids = _get_top_3_video_ids(recipe_name)
    summarys = []

    for video_id in video_ids:
        summarys.append(_generate_recipe_summary(recipe_name, video_id))
    return summarys