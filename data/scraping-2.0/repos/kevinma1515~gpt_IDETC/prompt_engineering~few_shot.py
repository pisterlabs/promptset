import openai
from config import API
from pathlib import Path
import csv

design_prompts_base = [
    "Generate 100 design solutions for a lightweight exercise device that can be used while traveling",
    "Generate 100 design solutions for a lightweight exercise device that can collect energy from human motion",
    "Generate 100 design solutions for a new way to measure the passage of time",
    "Generate 100 design solutions for a device that disperses a light coating of a powdered substance over a surface",
    "Generate 100 design solutions for a device that allows people to get a book that is out of reach",
    "Generate 100 design solutions for an innovative product to froth milk",
    "Generate 100 design solutions for a way to minimize accidents from people walking and texting on a cell phone",
    "Generate 100 design solutions for a device to fold washcloths, hand towels, and small bath towels",
    "Generate 100 design solutions for a way to make drinking fountains accessible for all people",
    "Generate 100 design solutions for a measuring cup for the blind",
    "Generate 100 design solutions for a device to immobilize a human joint",
    "Generate 100 design solutions for a device to remove the shell from a peanut in areas with no electricity",
    "Generate 100 design solutions for a device that can help a home conserve energy"
]


def load_human_responses():
    human_prompt_files = [
        "../sentence_embeddings/data/amazonTurkDesPrompt1.csv",
        "../sentence_embeddings/data/amazonTurkDesPrompt2.csv",
        "../sentence_embeddings/data/amazonTurkDesPrompt3.csv",
        "../sentence_embeddings/data/amazonTurkDesPrompt4.csv",
        "../sentence_embeddings/data/amazonTurkDesPrompt5.csv",
        "../sentence_embeddings/data/amazonTurkDesPrompt6.csv",
        "../sentence_embeddings/data/amazonTurkDesPrompt7.csv",
        "../sentence_embeddings/data/amazonTurkDesPrompt8.csv",
        "../sentence_embeddings/data/amazonTurkDesPrompt9.csv",
        "../sentence_embeddings/data/amazonTurkDesPrompt10.csv",
        "../sentence_embeddings/data/amazonTurkDesPrompt11.csv",
        "../sentence_embeddings/data/amazonTurkDesPromptNA.csv",
    ]

    human_prompts = []
    for human_prompt in human_prompt_files:
        prompts = []
        with open(human_prompt, 'r') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                prompts.append(row[0])
        prompts = [i.replace('\n', ' ') for i in prompts]
        human_prompts.append(prompts)

    return human_prompts


def craft_gpt_prompts(base_prompts, few_shot_examples=1):
    human_responses = load_human_responses()

    few_shot_prompts_all = []
    for design_prompt_count, design_prompt in enumerate(base_prompts):
        few_shot_prompts = []
        for example_count in range(few_shot_examples):
            human_example = human_responses[design_prompt_count][example_count]
            # design_prompt += f":\n 1. {human_example}"
            few_shot_prompts.append(design_prompt + f":\n 1. {human_example}")
        few_shot_prompts_all.append(few_shot_prompts)

    return few_shot_prompts_all


if __name__ == '__main__':
    openai.api_key = API
    few_shot_examples = 5   # Number of few shot examples that will be created for later averaging.

    design_prompts_few_shot = craft_gpt_prompts(design_prompts_base, few_shot_examples)

    for count_design, prompts in enumerate(design_prompts_few_shot):
        print(prompts)
        for count_prompt, prompt in enumerate(prompts):
            print(prompt)
            print("--------------------------------------------------")

            out = Path(f"data/few_shot/design_prompt_{count_design}_{count_prompt}.csv")
            if not out.exists():
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt,
                    temperature=0.9,
                    max_tokens=4097-len(prompt),
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )

                result = response['choices'][0]['text']
                print(result)

                out.parent.mkdir(parents=True, exist_ok=True)
                print(f"Saving to {out}")

                with out.open('w', newline='\n', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows([[x] for x in result.split("\n")])
            else:
                print(f"Prompt exists already")
