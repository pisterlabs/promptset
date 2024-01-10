import argparse

argparser = argparse.ArgumentParser(description="Ask a question and you shall receive an answer.")
argparser.add_argument(
    "--query",
    type=str,
    default="Where is Evan Fellman studying?",
    help="The question to ask",
)
argparser.add_argument(
    "--local",
    action="store_true",
    help="Use offline OPT IML 7B instead of ChatGPT",
)
argparser.add_argument(
    "--threshold",
    type=float,
    default=1e-3,
    help="The threshold us to consider the QA model's response as a confident answer",
)
args = argparser.parse_args()

from transformers import pipeline
import sys
import requests
import os
import openai


chatgpt = not args.local
result = None
result_idx = 0

pipe = pipeline("question-answering", model="deepset/roberta-base-squad2")
# Example call:
# pipe({'question': 'Evan Fellman height', 'context': 'I live in Boca Raton and am 23 years old'})
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_bing_result(query="cmu"):
    global result, result_idx
    subscription_key = os.getenv("BING_API_KEY")

    try:
        response = requests.get(
            "https://api.bing.microsoft.com/v7.0/search",
            headers={"Ocp-Apim-Subscription-Key": subscription_key},
            params={"q": query, "mkt": "en-US"},
        )
        response.raise_for_status()
    except Exception:
        pass

    result = response.json()["webPages"]["value"]
    result_idx = 0
    return result[result_idx]["snippet"]


def next_bing_result():
    global result, result_idx
    result_idx += 1
    if result_idx == len(result):
        return None
    return result[result_idx]["snippet"]


# Initialize the model
generator = None
if chatgpt:

    def generatorFunc(
        prompt,
        max_length,
        temperature,
        do_sample,
        top_k,
        top_p,
        repetition_penalty,
        num_return_sequences,
    ):
        return [
            {
                "generated_text": openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                )["choices"][0]["message"]["content"]
            }
        ]

    generator = generatorFunc
else:
    generator = pipeline("text-generation", model="facebook/opt-iml-max-1.3b")

notes = []
ready = False
while not ready:
    prompt = ""
    if len(notes) == 0:
        with open("start_prompt.txt", "r") as f:
            prompt = f.read().format(question=args.query)
    else:
        with open("get_query_with_notes_prompt.txt", "r") as f:
            prompt = f.read().format(question=args.query, notes="\n".join(notes))
    response = generator(
        prompt,
        max_length=200,
        temperature=0.75,
        do_sample=True,
        top_k=100,
        top_p=0.95,
        repetition_penalty=1,
        num_return_sequences=1,
    )[0]["generated_text"]
    query = response.split("\n")[-1].split("Query:")[-1].strip()

    print(f"LLM Query: {query}")
    answer_to_query = get_bing_result(query)

    print(f"  Answer: {answer_to_query}")
    content = False
    query_answer = pipe({"question": query, "context": answer_to_query})
    while not content:
        print()
        # using pretrained QA model to validate Bing results
        if query_answer["score"] >= args.threshold:
            print("  Content!")
            content = True
            break
        else:
            print("  Not content yet...")
            answer_to_query = next_bing_result()
            if answer_to_query is None:
                print("  No more Bing results!")
                break
            query_answer = pipe({"question": query, "context": answer_to_query})
            print(f"  Response: {answer_to_query}")

    notes.append(answer_to_query)
    answer_to_question = pipe({"question": args.query, "context": "\n".join(notes)})
    if answer_to_question["score"] >= args.threshold:
        print(f'Answer: {answer_to_question["answer"]}')
        exit()
