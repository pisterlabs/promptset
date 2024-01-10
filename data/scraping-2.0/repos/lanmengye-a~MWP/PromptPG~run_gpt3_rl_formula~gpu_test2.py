import argparse

import openai
import re
openai.api_key = "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"
train_sample =[{
"question":"Bryan took a look at his books as well . If Bryan has number0 books in each of his number1 bookshelves , how many books does he have in total ?",
"answer":"Answer = number0  * number1"
},
    {
        "question":"For the fifth grade play , the chairs have been put into number0 rows with number1 chairs in each row . How many chairs have been put out for the play ?",
        "answer":" Answer=(number0-number1)*number2"

    }
]
algebraicPrompt = "Write a mathematical equation and generate the answer format starting with `Answer ='"
def get_gpt3_output(que, args):
    prompt = "\n\n".join([f"Question: {sample['question']}\nAnswer: {sample['answer']}" for sample in
                           train_sample]) + "\n\n" +que+ "\n\n" + algebraicPrompt
    # message = [{"role": "user", "content": prompt1}]

    patience = 2
    while True:
        try:
            response = openai.Completion.create(engine=args.engine,
                                                prompt=prompt,
                                                temperature=args.temperature,
                                                max_tokens=args.max_tokens,
                                                top_p=args.top_p,
                                                frequency_penalty=args.frequency_penalty,
                                                presence_penalty=args.presence_penalty,
                                                stop=["\n"])
            output = response["choices"][0]["text"].strip()
            break
        except Exception as e:
            patience -= 1
            if not patience:
                print("!!! Running out of patience waiting for OpenAI")
            else:
                print(e)
                time.sleep(0.1)
    return output

if __name__ =="__init__":
    # GPT-3 settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='text-davinci-002', choices=['text-davinci-002', 'ada'])
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    args = parser.parse_args()
    que="Frank made number0 dollars mowing lawns over the summer . If he spent number1 dollars buying new mower blades , how many number2 dollar games could he buy with the money he had left ?"
    get_gpt3_output(que,args)

