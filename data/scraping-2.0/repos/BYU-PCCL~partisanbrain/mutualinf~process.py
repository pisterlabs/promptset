from anes_backstory import make_backstory1, make_backstory2, make_backstory3
from collections import defaultdict

import numpy as np
import openai
import os
import tqdm

# Authenticate with openai
openai.api_key = os.getenv("GPT_3_API_KEY")


TEMPLATES = {
    "template_1a": lambda row: f"{make_backstory1(row)}In 2016, between Clinton and Trump, I voted for",
    "template_1b": lambda row: f"{make_backstory1(row)}In 2016 the candidate I voted for was",
    "template_1c": lambda row: f"{make_backstory1(row)}If asked to choose either \"Clinton\" or \"Trump\" in response to the question \"Did you vote for Clinton or Trump in the 2016 election\" I would choose \"",
    "template_1d": lambda row: f"{make_backstory1(row)}Today someone asked me whether I voted for Clinton or Trump in the 2016 election, and I responded that I voted for",
    "template_1e": lambda row: f"{make_backstory1(row)}In 2016, Hillary Clinton and Donald Trump ran for president. In the election, I voted for",
    "template_2a": lambda row: f"{make_backstory2(row)}In 2016, between Clinton and Trump, I voted for",
    "template_2b": lambda row: f"{make_backstory2(row)}In 2016 the candidate I voted for was",
    "template_2c": lambda row: f"{make_backstory2(row)}If asked to choose either \"Clinton\" or \"Trump\" in response to the question \"Did you vote for Clinton or Trump in the 2016 election\" I would choose \"",
    "template_2d": lambda row: f"{make_backstory2(row)}Today someone asked me whether I voted for Clinton or Trump in the 2016 election, and I responded that I voted for",
    "template_2e": lambda row: f"{make_backstory2(row)}In 2016, Hillary Clinton and Donald Trump ran for president. In the election, I voted for",
    "template_3a": lambda row: f"{make_backstory3(row)}In 2016, between Clinton and Trump, I voted for",
    "template_3b": lambda row: f"{make_backstory3(row)}In 2016 the candidate I voted for was",
    "template_3c": lambda row: f"{make_backstory3(row)}If asked to choose either \"Clinton\" or \"Trump\" in response to the question \"Did you vote for Clinton or Trump in the 2016 election\" I would choose \"",
    "template_3d": lambda row: f"{make_backstory3(row)}Today someone asked me whether I voted for Clinton or Trump in the 2016 election, and I responded that I voted for",
    "template_3e": lambda row: f"{make_backstory3(row)}In 2016, Hillary Clinton and Donald Trump ran for president. In the election, I voted for",
    "template_3f": lambda row: f"{make_backstory3(row)}Q: In 2016, between Clinton and Trump, who did you vote for?\n\nA:",
    "template_3g": lambda row: f"{make_backstory3(row)}Q: In 2016, between Clinton and Trump, who did you vote for?\n\nA: I voted for",
    "template_3h": lambda row: f"{make_backstory3(row)}Q: Who did you vote for in the 2016 election? Clinton or Trump?\n\nA:",
    "template_3i": lambda row: f"{make_backstory3(row)}Q: Who did you vote for in the 2016 election?\n\nA:",
    "template_3j": lambda row: f"{make_backstory3(row)}Q: In 2016, Hillary Clinton and Donald Trump ran for president. Who did you vote for - Clinton or Trump?\n\nA:",
}


def make_prompt(row, template_name):
    return f"{TEMPLATES[template_name](row)}"


def process_prompt(prompt, gpt_3_engine):
    assert gpt_3_engine in ["ada", "babbage", "curie", "davinci"]
    return openai.Completion.create(engine=gpt_3_engine,
                                    prompt=prompt,
                                    max_tokens=1,
                                    logprobs=100)


def extract_category_probs(gpt_3_response, categories_key_sets):
    probs_dict = gpt_3_response["choices"][0]["logprobs"]["top_logprobs"][0]

    # Do softmax
    probs_dict_exp = {k: np.exp(v) for (k, v) in probs_dict.items()}

    score_dict = {k: 1e-6 for k in categories_key_sets.keys()}

    # Only take the keys from probs_dict_exp that match allowed values
    # and aggregate
    for prob_key, prob in probs_dict_exp.items():
        prob_key = prob_key.strip().lower()
        if len(prob_key):
            for category_name in score_dict:
                for category_key in categories_key_sets[category_name]:
                    if category_key.strip().lower().startswith(prob_key):
                        score_dict[category_name] += prob

    return score_dict


def process_df(df,
               gpt_3_engine,
               template_names,
               dv_col_name,
               categories_key_sets=None):

    new_df_dict = defaultdict(list)

    if categories_key_sets is None:
        categories = df[dv_col_name].unique().tolist()
        categories_key_sets = dict(zip(categories, [[c] for c in categories]))
    else:
        categories = list(categories_key_sets.keys())

    for _, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

        for template_name in template_names:

            new_df_dict["template_name"].append(template_name)

            prompt = make_prompt(row, template_name)
            new_df_dict["prompt"].append(prompt)
            new_df_dict["categories"].append(categories)
            new_df_dict["ground_truth"].append(row[dv_col_name])

            response = process_prompt(prompt, gpt_3_engine)
            category_probs = extract_category_probs(response,
                                                    categories_key_sets)

            coverage = sum(category_probs.values())

            # Normalize category_probs
            category_probs = {k: v / coverage
                              for (k, v) in category_probs.items()}

            try:
                response = process_prompt(prompt, gpt_3_engine)
                new_df_dict["response"].append(response)
            except Exception as e:
                print("Exception in process_df:", e)
                new_df_dict["response"].append(None)

            new_df_dict["coverage"].append(coverage)

            for category in categories:
                new_df_dict[category].append(category_probs[category])

    return pd.DataFrame(new_df_dict)


if __name__ == "__main__":
    import pandas as pd
    # toy_df = pd.DataFrame(dict(
    #     age=["male", "male", "female"],
    #     food=["vegan", "a meat lover", "vegan"],
    #     dv=["carrots", "bacon", "carrots"]
    # ))

    # df = process_df(toy_df, "davinci", TEMPLATES.keys(), "dv", {"bacon": ["bacon", "meat"], "carrots": ["carrots"]})
    # df.to_csv("temp.csv", index=False)

    df = pd.read_csv("formatted_anes.csv")
    df = df.dropna()
    df = df.sample(n=200)

    out = process_df(df, "davinci", TEMPLATES.keys(), "2016_presidential_vote",
                     {"Hillary Clinton": ["Hillary", "Clinton"],
                      "Donald Trump": ["Donald", "Trump"]})
    out.to_csv("exp_results.csv", index=False)
