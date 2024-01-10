import openai 
from typing import List


# Text generation

def ask_chatgpt(message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
    )
    return response["choices"][0]["message"]["content"]

            
def assist_journalist(
    facts: List[str], tone: str, length_words: int, style: str
):
    prompt_role = "You are an assistant for journalists. \
        Your task is to write articles based on the FACTS taht are given to you. \
        You should respect the insructions: the TONE, the LENGTH, and the STYLE"

    facts = ", ".join(facts)
    prompt = f"{prompt_role} \
        FACTS: {facts} \
        TONE: {tone} \
        LENGTH: {length_words} words \
        STYLE: {style}"
    return ask_chatgpt([{"role": "user", "content": prompt}])

def example_call():
    print(assist_journalist( 
          facts=[
              "A book on ChatGPT has been published last week",
              "The title is Developing Apps with GPT-4 and ChatGTP",
              "The publisher is O'Reilly.",
          ],
          tone="excited",
          length_words=50,
          style="news flash",
          )
    )
    
# Summarizing YouTube videos

def summarize_youtube_transcript(transcript):
    response = openai.ChatCompletion.create(
        model="gp3-5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "Summarize the following text."},
            {"role": "assistant", "content": "Yes."},
            {"role": "user", "content": transcript},
        ],
    )
    print(response["choices"][0]["message"]["content"])
    
# Note: if the transcript is too long, you can break it into chunks and get them summarized separately.
# And then you can combine the summaries into one and get it summarized again.



