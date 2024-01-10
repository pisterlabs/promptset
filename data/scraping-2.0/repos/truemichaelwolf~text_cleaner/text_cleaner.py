import os
import openai
import tiktoken
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

def get_response(prompt, model="gpt-3.5-turbo", print_response=False):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )

    if print_response:
        print(response.choices[0].message["content"])
    return response.choices[0].message["content"]

def make_refine_text_prompt(text):
    return text_refine_msg + f"####\n{text}\n####"

def refine_text(text, model="gpt-3.5-turbo"):
    prompt = make_refine_text_prompt(text)
    return get_response(prompt, model=model, print_response=False)

if __name__ == "__main__":

  text_refine_msg = """
  I want you to refine texts extracted from a PDF file with an OCR model. The original PDF is an scanned English Test Paper.
  It main contain the following mistakes:
  - Missing spaces between words, e.g., "Iwantyou" instead of "I want you"
  - Missing spaces between sentences, e.g., "I want you.Do you want me?" instead of "I want you. Do you want me?"
  - wrong spelling, e.g., "temible" instead of "terrible"
  - Random symbols, e.g., "Humans are taller and taller, wn - Stale hocame emall-— 5° : fact’" instead of "Humans are taller and taller, while some animals have become smaller. In fact"
  - Unnecessary line breaks
  - Unnecessary page breaks
  - Unnecessary texts in the header or footer
  - Any other mistakes you can think of
  Note that the text may contain both Chinese and English, but DO NOT TRANSLATE. Your answer should contain ONLY THE REFINED TEXT, NOTHING ELSE.
  I will provide the text below between two lines of four # symbols. Please refine the text using your best judgement. 
  """

  with open("text_b4_refinement.txt", "r", encoding="utf-8") as f:
    text = f.read()


  refined_text = refine_text(text, model="gpt-4")
