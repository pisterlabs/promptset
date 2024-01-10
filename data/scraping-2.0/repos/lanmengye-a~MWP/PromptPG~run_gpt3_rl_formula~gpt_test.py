import argparse

import openai
import time
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
    prompt1 = "\n\n".join([f"Question: {sample['question']}\nAnswer: {sample['answer']}" for sample in
                           train_sample]) + "\n\n" +que+ "\n\n" + algebraicPrompt
    message = [{"role": "user", "content": prompt1}]

    params = {
        "model": args.model,
        "max_tokens": 2048,
        "temperature": args.temperature,
        "messages": message
    }
    for retry in range(3):
        try:
            response = openai.ChatCompletion.create(**params)["choices"][0]["message"]["content"]
            message.append({"role": "assistant", "content": response})
            tool = "\n\n".join(re.findall(r"```python\n(.*?)```", response, re.DOTALL))
            # exec(tool, globals())
            break
        except Exception as e:
            print("ERROR: failed to generate tool", e)
            message.append({"role": "user",
                            "content": f"Failed to execute the function due to the error: {type(e).__name__} {e}. Please fix it and try again."})

    print("Tool:", message[-1]["content"])

    message.append({"role": "assistant", "content": response})

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

