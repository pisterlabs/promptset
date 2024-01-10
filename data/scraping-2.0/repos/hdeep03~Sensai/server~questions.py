import openai
from transformers import GPT2TokenizerFast
import os
KEY=os.getenv("KEY")

openai.api_key = os.environ["OPENAI_API_KEY"]

def answer_question(question):
    prompt = f"Explain this to me as if I am a college student: {question}"
    model_engine = "text-davinci-003"
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    encoded = tokenizer.encode(prompt)
    numberOfBatchTokens = len(encoded)

    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens= 4097 - numberOfBatchTokens,
        n=1,
        stop=None,
        temperature=0.5,
    )
    response = completion.choices[0].text
    return response