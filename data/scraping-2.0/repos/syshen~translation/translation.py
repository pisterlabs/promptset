import requests
import openai
import torch
import streamlit as st
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def count_tokens(text):
  input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
  num_tokens = input_ids.shape[1]
  return num_tokens

def text_to_chunks(text, chunk_size=2000):
  punctuation = '.!?'

  sentences = []
  start = 0
  for i, char in enumerate(text):
    if char in punctuation:
      sentences.append(text[start:i+1])
      start = i+1
  if i < len(text):
    sentences.append(text[start:])

  chunks = []
  chunk = None
  current_chunk_size = 0
  for i, sentence in enumerate(sentences):
    tokens = tokenizer.encode(sentence)
    num_tokens = len(tokens)
    if (current_chunk_size + num_tokens) >= chunk_size:
      current_chunk_size = num_tokens
      chunks.append(chunk)
      chunk = tokens
    else:
      if chunk is None:
        chunk = tokens
      else:
        chunk.extend(tokens)
      current_chunk_size += num_tokens
  chunks.append(chunk)

  return chunks

def callOpenAI(prompt_request, max_tokens=500):
  try:
    response = openai.Completion.create(
              model="text-davinci-003",
              prompt=prompt_request,
              temperature=.5,
              max_tokens=max_tokens,
              top_p=1,
              frequency_penalty=0,
              presence_penalty=0
    )
    return response.choices[0].text
  except:
    st.error("Error", icon="🚨")

@st.cache_data
def translate_text(translating):
  chunks = text_to_chunks(translating, chunk_size=500)
  translated = []

  for i, chunk in enumerate(chunks):
    text = tokenizer.decode(chunks[i])
    result = callOpenAI(f"Translate to Traditional Chinese: {text}", max_tokens=3500)
    st.write(result)
    translated.append(result)    

  return ''.join(translated)


st.header("AI 中文翻譯")
st.write("""
因為需要使用 OpenAI 進行中文翻譯，所以請自行前往 OpenAI 網站申請 API Key，並在此輸入所取得的 API Key。
""")
openai.api_key = st.text_input("OpenAI API Key")

text = st.text_area("輸入需要翻譯的文章內容", height=600)
if text:
  with st.spinner("翻譯小編努力中 ..."):
    translation = translate_text(text)
    translation
