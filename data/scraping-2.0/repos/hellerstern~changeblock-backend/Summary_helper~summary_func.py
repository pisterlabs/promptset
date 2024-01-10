import openai
import os
from transformers import AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def summary(text):
    SUMMARY_TEMPLATE = "Summarize the following text: {text}"
    prompt = SUMMARY_TEMPLATE.format(text=text)
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt= "Summarize the following \n" + str(text),
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response.choices[0].text

def summarize(text, size=2800, mean_tokens=2000):
    # split text into many parts
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokens = tokenizer.encode(text)
    parts = [tokenizer.decode(tokens[i:i+size]) for i in range(0, len(tokens), size)]
    print('Number of parts:', len(parts))
    # call OpenAI API for each part
    text_sum = [summary(part) for part in parts]
    text_sum = '\n'.join(text_sum)
    if len(tokenizer.encode(text_sum)) > mean_tokens:
        summarize(text_sum, size)
    else:
        return text_sum