import re
import pandas as pd
from typing import Dict
from datasets import load_dataset
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from tqdm import tqdm


def infer(prompt: str, option1: str, option2: str, correct: str) -> Dict:
    # chain-of-thought prompt
    winogrande_prompt = PromptTemplate.from_template(
        """
        John moved the couch from the garage to the backyard to create space. The _ is small. "garage" or "backyard"?
        A: The question is asking about what is small, the backyard or the garage. Common sense says that space needed to be created in the garage because it was perceived as not big enough, or small. The answer is "garage".
        The doctor diagnosed Justin with bipolar and Robert with anxiety. _ had terrible nerves recently. "Justin" or "Robert"?
        A: The question is asking about who has terrible nerves, Justin or Robert. Common sense says that people who has anxiety are nervous, so Robert who was diagnosed with anxiety is likely to have terrible nerves. The answer is "Robert".
        Dennis drew up a business proposal to present to Logan because _ wants his investment. "Dennis" or "Logan"?
        A: The question is asking about who wants the investment, Logan or Dennis. Common sense says that the person who make the proposal - Dennis - wants something from the other person. The answer is "Dennis".
        Felicia unexpectedly made fried eggs for breakfast in the morning for Katrina and now _ owes a favor. "Felicia" or "Katrina"?
        A: The question is asking about who owes a favor, Katrina or Felicia. Common sense says that the person who received something - Katrina - and hasn't done anything in return owes a favor. The answer is "Katrina".
        My shampoo did not lather easily on my Afro hair because the _ is too dirty. "shampoo" or "hair"?
        A: The question is asking about what is dirty, the hair or the shampoo. Common sense says that dirty hair, especially Afro hair, is hard and stiff, so it can be hard to lather shampoo on. The answer is "hair".
        {prompt} "{option1}" or "{option2}"?
        A: 
        """
    )
    prompt_filled = winogrande_prompt.format(prompt=prompt, option1=option1, option2=option2)
    model_output = llm(prompt_filled)
    stripped = str(model_output).strip()
    option1_regex = f"The answer is \"{option1}\""
    option2_regex = f"The answer is \"{option2}\""
    if re.search(option1_regex, stripped, re.IGNORECASE):
        answer = "1"
    elif re.search(option2_regex, stripped, re.IGNORECASE):
        answer = "2"
    else:
        print(f"[ERROR]: {stripped}")
        answer = "0"

    output_dict = {
        "prompt": prompt,
        "option1": option1,
        "option2": option2,
        "model_output": model_output,
        "stripped": stripped,
        "answer": answer,
        "correct": correct,
        "correctness": answer == correct,
    }
    return output_dict

if __name__ == "__main__":
    model = "mistral"

    dataset = load_dataset("winogrande", "winogrande_debiased")
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    llm = Ollama(model=model, temperature=0.8, top_p=0.9, top_k=0)

    answers = []
    for idx in tqdm(range(100)):
        # replace all "_" with "[BLANK]"
        prompt = val_dataset["sentence"][idx]
        option1 = val_dataset["option1"][idx]
        option2 = val_dataset["option2"][idx]
        correct_ans = val_dataset["answer"][idx]
        # print(f"[MAIN]: create and start thread {idx}")
        answers.append(infer(prompt, option1, option2, correct_ans))


    # get average correctness
    correctness = [answer["correctness"] for answer in answers]
    print(f"Average correctness: {sum(correctness) / len(correctness)}")

    # save answers to csv with pandas
    output_df = pd.DataFrame.from_records(answers)
    output_df.to_csv("./output/{}-{}-sample-{}-accuracy.csv".format(
        model, 
        len(answers), 
        sum(correctness) / len(correctness)
    ), index=False)
        