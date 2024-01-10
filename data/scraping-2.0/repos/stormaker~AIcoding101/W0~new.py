import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
openai_api = os.getenv("OPENAI_API_KEY")

# Step 1: Read the text file
with open("input.txt", "r", encoding='utf-8') as f:
    text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1100,
    chunk_overlap  = 20,
    length_function = len,
    is_separator_regex = False,

)
chunks = text_splitter.split_text(text)
#print(chunks[0])

client = OpenAI()
def summarize_text(text):

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "assistant",
                "content": f"你是会议记录整理人员，以下是一段录音的逐字稿，请逐字将其整理成前后连贯的文字，需要注意：1.保留完整保留原始录音的所有细节。2.尽量保留原文语义、语感。3.请修改错别字，符合中文语法规范。4.去掉说话人和时间戳。5.第一人称：我。6.请足够详细，字数越多越好。7.保持原始录音逐字稿的语言风格:{text}"
            }
        ],
        temperature=0,
        max_tokens=4000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content


print("Summarizing the text...")
summary = []
summaries = [summarize_text(chunk) for chunk in chunks]

# Step 4: Combine the result from the summarize chain and save them as a new text file
with open('summary_output.txt', 'w', encoding='utf-8') as file:
    for summary in summaries:
        file.write(summary + '\n\n')

print("Summarization complete. Check the summary_output.txt file.")
