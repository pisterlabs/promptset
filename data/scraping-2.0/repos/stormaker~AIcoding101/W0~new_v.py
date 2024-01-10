import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

load_dotenv()
openai_api = os.getenv("OPENAI_API_KEY")

# Step 1: Read the text file
with open("input.txt", "r", encoding='utf-8') as f:
    text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_text(text)

client = OpenAI()


def summarize_text(text):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "assistant",
                "content": f"请将我发送的语音稿整理为正式的书面语，要求"
                           f"-保留对话中的每一个细节，不改变原文含义；"
                           f"-尽可能地保留原话的用词、话语风格；"
                           f"-请修改错别字，符合中文语法规范。"
                           f"-去掉说话人和时间戳。:{text}"
            }
        ],
        temperature=0,
        max_tokens=4000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content


print("Summarizing the text...\n")
summaries = []
with tqdm(total=len(chunks)) as pbar:
    for i, chunk in enumerate(chunks):
        summary = summarize_text(chunk)
        summaries.append(summary)
        # Save each summary chunk in a separate text file
        with open(f'summary_chunk_{i+1}.txt', 'w', encoding='utf-8') as file:
            file.write(summary)
        pbar.update(1)

# Combine the summary chunk files into one text file
with open('summary_output.txt', 'w', encoding='utf-8') as file:
    for i, summary in enumerate(summaries):
        # file.write(f'Summary Chunk {i+1}:\n\n')
        file.write(summary + '\n\n')

print("Summarization complete. Check the summary_output.txt file.")