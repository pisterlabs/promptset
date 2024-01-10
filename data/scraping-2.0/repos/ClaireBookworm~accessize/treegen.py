import json
import openai
import tiktoken
import os
import dotenv
dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL = "text-davinci-003"
TOKEN_CAPACITY = 4000


def run_gpt(prompt: str, max_tokens: int):
    response = openai.Completion.create(
        model=MODEL,
        prompt=prompt,
        temperature=0,
        max_tokens=max_tokens,
    )
    return response.choices[0].text

SUMMARIZE_PROMPT = """Write a {paragraphs} paragraph the following about accessibility guidelines into readable concise clear prose without any special character, focusing on what a developer would need to specifically implement while ignoring section titles and numbers:

Text: {text}

Summary of {paragraphs} paragraphs:"""
SUMMARIZE_PROMPT_TOKENS = len(tiktoken.get_encoding("cl100k_base").encode(SUMMARIZE_PROMPT))


def summarize(s: str, max_tokens: int) -> str:
    return run_gpt(SUMMARIZE_PROMPT.format(text=s, paragraphs=max_tokens//100), max_tokens)


def recursive_summarize(tree) -> str:
  max_tokens = (TOKEN_CAPACITY // (2*(1 + len(tree["children"])))) - SUMMARIZE_PROMPT_TOKENS
  # assert (SUMMARIZE_PROMPT_TOKENS + max_tokens) * (1 + len(tree["children"])) < TOKEN_CAPACITY//2, f"Token capacity exceeded. Max tokens: {max_tokens} for {len(tree['children'])}"

  summaries = [summarize(tree['section_text'] , max_tokens) if 'section_text' in tree else '']
  for child in tree["children"]:
      recursive_summarize(child)
  prompt = ''.join(summaries)
  print('PROMPT:\n', prompt)
  print('PROMPT TOKENS:', len(tiktoken.get_encoding("cl100k_base").encode(prompt)), 'out of', max_tokens)
  tree['summary'] = summarize(prompt, max_tokens)
  print(tree['summary'])


if __name__ == '__main__':
  with open('toc.txt', 'r') as f:
    inp = f.read()

  titles = []
  for name in inp.split("\n"):
    titles.append(name.strip())

  with open('wcag21_embeddings.json', 'r') as f:
    data = json.load(f)
  
  recursive_summarize(data)
  with open('wcag21_embeddings_summaries.json', 'w') as f:
    json.dump(data, f, indent=2)

