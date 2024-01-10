import openai
import os

API_KEY = os.environ.get('OpenaiAPI_KEY')
if not API_KEY:
    print("No API KEY found")
    exit()
openai.api_key = API_KEY

if __name__ == '__main__':
    prompt_f = open("prompts.json")
    import json

    prompts_info = json.load(prompt_f)
    for prompt_info in prompts_info:
        print(prompt_info["id"], ': ', prompt_info["prompt"])
    id_prompt = input("Choose a prompt by its id: ")
    prompt = ""
    if int(id_prompt) in [i + 1 for i in range(len(prompts_info))]:
        prompt = prompts_info[int(id_prompt) - 1]["prompt"]
    else:
        print("Please input a id in range ", [i + 1 for i in range(len(prompts_info))])

    file = open("../data/jointslu/sentences.txt")
    sentences = file.readlines()
    index = 0
    while True:
        sentence = sentences[index].replace('\n', '')
        index = index + 1
        prompt = prompt + '\n' + "instruction:\n" + sentence
        response = openai.Completion.create(engine="text-davinci-001",
                                            prompt=prompt,
                                            max_tokens=256)
        print("\n=====================")
        print(sentence)
        print(f"response {response}")
        k = input("Press any key to continue, press 'q' to quit. ")
        if k == 'q':
            break
