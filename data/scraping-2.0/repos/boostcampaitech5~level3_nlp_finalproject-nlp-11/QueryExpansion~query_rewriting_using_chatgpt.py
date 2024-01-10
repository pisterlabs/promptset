import openai
import json
import argparse
import re
import os
openai.api_key = os.getenv("OPENAI_API_KEY")  # Replace with your OpenAI API key
system_message = f"You are very helpful assistant."
                   
def rephrase_sentence(sentence):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": sentence}
        ]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

def main(args):
    #file read
    file_path = "./all_questions.json"
    p = re.compile(r"^\s*\d+\.*\s*\:*\s*")
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    #Query sentence size 100 per epoch
    for idx in range(args.start_data_index, args.end_data_index, 100):
        query = "paraphrase below 100 questions to look more natural.\n"
        sentence = query + '\n'.join(f'{ii} : {vv},' for ii, vv in enumerate(data[idx:idx + 100], 1))
        rephrased_sentence = rephrase_sentence(sentence)
        rephrased_sentence = rephrased_sentence.splitlines()
        
        # If unable to generate 100 queries perfectly, continue executing until they are generated
        while len(rephrased_sentence) != 100:
            print(f"returned message less than 100, stopped idx : {idx}")
            rephrased_sentence = rephrase_sentence(sentence)
            rephrased_sentence = rephrased_sentence.splitlines()
        rephrased_sentence = list(map(lambda x: re.sub(p, "", x), rephrased_sentence))
        
        #file write
        with open('paraphrased_sentences.txt','a',encoding='utf-8') as f:
            for i, v in enumerate(rephrased_sentence):
                f.write(f"{idx + i} ::: {v}")
                f.write("\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--start_data_index", type=int, default=0, help="data selection")
    parser.add_argument("--end_data_index", type=int, default=0, help="data selection")
    args = parser.parse_args()
    main(args)