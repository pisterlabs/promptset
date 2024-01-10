import openai
import json
from tqdm import tqdm
import time
import argparse
import multiprocessing

f = open("/home/shang/openai-apikey.txt")
#print(f.readlines()[0])
openai.api_key = f.readlines()[0]
f.close()

def first_pass():
    f = open("./COIN/base/coin_categories.json")
    data = json.load(f)
    category_steps = {}
    with multiprocessing.Pool(10) as p:
        for result in p.starmap(get_category, [(category, data) for category in data.keys()]):
            category_steps[result[1]] = result[0]
    with open("./COIN/base/category_base_steps.json", "w") as outfile:
        json.dump(category_steps, outfile)
    return category_steps

def get_category(category, data):
    ids = data[category]["ids"].split(",")
    steps = {}
    prompt = "Write the steps of the task that the person is demonstrating, based on the noisy transcript.\nTranscript: |||1\nSteps:\n1."
    for i in tqdm(range(len(ids))):
        try:
            f = open("/shared/medhini/COIN/coin_asr/" + ids[i] + ".txt")
            transcript = " ".join(f.readlines())
            #print(transcript)
            input_text = prompt.replace("|||1", transcript)
            tries, max_attempts = 0, 5
            while tries < max_attempts:
                time.sleep(0.5)
                try:
                    response = openai.Completion.create(
                                            engine="text-babbage-001",
                                            prompt=input_text,
                                            temperature=0.7,
                                            max_tokens=256,
                                            top_p=1,
                                            frequency_penalty=0,
                                            presence_penalty=0
                                        )
                except Exception as exc:
                    print(f"OPENAI ERROR, try ({tries}), {exc}")
                tries += 1
            output = response["choices"][0]["text"].strip()
            steps[ids[i]] = output
        except:
            print(ids[i])
            pass
    return steps, category

def second_pass(category_steps):
    if args.no_fp:
        with open("./COIN/base/category_base_steps.json") as r:
            category_steps = json.load(r)
    with open("./COIN/base/coin_categories.json") as file:
        data = json.load(file)
    for category in category_steps:
        print(f"CATEGORY: {category}")
        steps = []
        for id in category_steps[category]:
            steps.append(category_steps[category][id])
        input_message=[
                {"role": "system", "content": f"Extract a set of concise general steps for perfoming the task: {category} from the following recipes. Be as vague and general as possible. For your output, only include the steps without extra information."},
            ]
        for step in steps:
            input_message.append({"role": "user", "content": step})
            input_message.append({"role": "assistant", "content": "Recieved, waiting on next step list."})
        input_message.append({"role": "user", "content": "I have inputted all recipes. Now, give me a general recipe like I instructed before."})
        response = openai.ChatCompletion.create(
          model="gpt-4",
          messages=input_message
        )
        #print(response)
        output = response["choices"][0]["message"]["content"].strip()
        print(output)
        print("============================")
        data[category]["general steps"] = output
    with open("./COIN/base/coin_gen_seps.json", "w") as outfile:
        json.dump(data, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_fp", action="store_true")
    args = parser.parse_args()

if not args.no_fp:
    steps = first_pass()
else:
    steps = None
second_pass(steps)
