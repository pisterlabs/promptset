import json
import os
import openai
import wandb
import pandas as pd
import time

openai.api_key = ""  # os.getenv("OPENAI_API_KEY")
# openai/.api_key = os.getenv("OPENAI_API_KEY")


# run = wandb.init(project='GPT-3 job description')
# prediction_table = wandb.Table(columns=["prompt", "completion"])

model = "text-davinci-003"
# model = "text-ada-001"


def make_completion_request(model, question):
    start_sequence = "\nA:"
    restart_sequence = "\n\nQ: "

    response = openai.Completion.create(
        model=model,
        # prompt=f"I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. "
        # f"If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with 'Unknown'.\n\nQ: What is human life"
        # f" expectancy in the United States?\nA: Human life expectancy in the United States is 78 years.\n\nQ: Who was president of the United"
        # f" States in 1955?\nA: Dwight D. Eisenhower was president of the United States in 1955.\n\nQ: Which party did he belong to?\nA: He "
        # f"belonged to the Republican Party.\n\nQ: What is the square root of banana?\nA: Unknown\n\nQ: How does a telescope work?\nA: "
        # f"Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\nQ: Where were the 1992 Olympics held?\nA: "
        # f"The 1992 Olympics were held in Barcelona, Spain.\n\nQ: How many squigs are in a bonk?\nA: Unknown\n\n"
        # f"Q: {question}?\nA: ",
        prompt=f"I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. "
        f"If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with 'Unknown'.\n\nQ: What is human life"
        f" expectancy in the United States?\nA: 78 years\n\nQ: Who was president of the United"
        f" States in 1955?\nA: Dwight D. Eisenhower\n\nQ: Which party did he belong to?\nA: "
        f"Republican Party\n\nQ: Where were the 1992 Olympics held?\nA: "
        f"Barcelona, Spain.\n\n"
        f"Q: {question}\nA: ",
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"],
    )

    return response


def get_gpt3_completions(model, data, ds_name):
    job_df = {"q_id": [], "question": [], "answer": []}

    print(data.head())
    for index in range(0, len(data)):
        question = data[index]

        try:
            if index % 20 == 0 and index != 0:
                print("************", index)
                time.sleep(40)
            response = make_completion_request(model, question)
        except:
            print("exception/sleep")
            time.sleep(60)
            response = make_completion_request(model, question)

        job_df["q_id"].append(index)
        job_df["question"].append(question)

        if response.choices[0].text.strip() == "":
            job_df["answer"].append("<no response>")
        else:
            job_df["answer"].append(response.choices[0].text.strip())
        final_df = pd.DataFrame(job_df)
        final_df.to_csv(f"gpt_{model}_predictions_{ds_name}.csv", index=False, sep="\t")
        # print(final_df)
    return final_df


def load_dataset(ds_path):
    data = pd.read_csv(ds_path)

    # TODO: Clean the __or__ occurrences and save in usable format for evaluation

    return data["Question"]


def main():
    # ds_path, ds_name = (
    #     "/home/wallat/temporal-llms/data/Event-focused Questions/Explicitly Time-Scoped Questions.csv",
    #     "explicit",
    # )
    ds_path, ds_name = (
        "/home/wallat/temporal-llms/data/Event-focused Questions/Implicitly Time-Scoped Questions.csv",
        "implicit",
    )

    data = load_dataset(ds_path)

    final_df = get_gpt3_completions(model, data, ds_name)
    print(final_df)


if __name__ == "__main__":
    main()
