import argparse
import os
import joblib
import openai
import pandas as pd
import time
from tqdm import tqdm
from config import api_key_to_use
# NOTE: api_key_to_use is a dictionary containing the api keys to use with keys being indices starting from 0 and values being the api keys


data_dir = "final_data"

prompts_dict = {
    "structure": "People either like coffee or hate it\nAnswer: People either like x or hate x.\n###\nIf you don't drive a car, you hate the Earth.\nAnswer: if you don't do x, you do y\n###\nALL teenagers are irresponsible\nAnswer: All Xs are y\n###\n\"Chick-Fil-A is the greatest food chain in existence. Look at the crowds!\"\nAnswer: X is the greatest y in existence. Look at the crowds.\n###\nHave you stopped cheating on exams?\nAnswer: Have you stopped doing x?\n###\n\"The two courses I took at UWI were not very interesting. I don't think its a good university.\"\nAnswer: The two Xs I took at Y were not very interesting. I don't think it's a good Y.\n###\nIf you are not with us, you are against us.\nAnswer: If you are not x, you are y.\n###\nI called my friend and he didn't answer. He wasn't home or he was dead.\nAnswer: I did x and he didn't y. He was z or he was a.\n###\nDr. Fauci says we need to wear masks, but Dr. Fauci is the oldest, meanest doctor around. I’m sure what he says isn’t true at all.\nAnswer:  X says we need to do y, but X is the oldest, meanest z around. I'm sure what they say isn't true at all.\n###\nAccording to the Supreme Court, we have a right to abortion. Therefore, it is right to have an abortion.\nAnswer: According to X, we have a right to y. Therefore, it is right to do y.\n###\n{}\nAnswer:",
    "explanations": "analyze the argument \"guys are messy and unclean, but are really hard workers\"\nIt's not appropriate to make generalizations about people based on their gender. The argument \"guys are messy and unclean, but are really hard workers\" suggests that all men are messy and unclean, which is not true. Additionally, the argument implies that being messy and unclean is at odds with being a hard worker, which is also not necessarily true. It's important to remember that every person is an individual with their own unique characteristics, and it's not fair to stereotype people or make assumptions about them based on their gender.\n###\nanalyze the argument \"Chick-Fil-A is the greatest food chain in existence. Look at the crowds!\"\nThe argument \"Chick-Fil-A is the greatest food chain in existence. Look at the crowds!\" is based on the premise that the popularity of a food chain is indicative of its quality. While it's true that Chick-Fil-A may be popular and have large crowds, this does not necessarily mean that it is the \"greatest\" food chain. There could be many factors that contribute to why a particular food chain is popular, such as location, price, or marketing efforts. It's important to consider a variety of factors when evaluating the quality of a food chain, rather than relying on popularity alone.\n###\nanalyze the argument \"If you are not with us, you are against us.\"\nThe argument \"If you are not with us, you are against us\" suggests that there are only two sides to an issue or situation, and that if you are not actively supporting one side, you are automatically opposed to it. This kind of black-and-white thinking can be problematic because it doesn't allow for the complexity and nuance that often exists in real-world situations. It can also create an \"us vs. them\" mentality that can lead to conflict and division. It's important to recognize that there may be multiple perspectives on an issue, and that it's possible to have a nuanced or balanced view that doesn't align with either side completely.\n###\nanalyze the argument \"The accident was caused by the taxi parking in the street\"\nThe argument \"The accident was caused by the taxi parking in the street\" suggests that the taxi parking in the street was the only factor that contributed to the accident. While it's true that the taxi parking in the street may have been a contributing factor, there could have been other factors that also contributed to the accident. For example, the driver of the car may have been driving too fast, or the driver of the taxi may have been distracted. It's important to consider all of the factors that may have contributed to the accident, rather than focusing on only one.\n###\nanalyze the argument \"Justin Beiber wears Ray Bans, so you should buy a pair to wear in the sun.\"\nThe argument \"Justin Beiber wears Ray Bans, so you should buy a pair to wear in the sun\" suggests that because Justin Beiber wears Ray Bans, they are the best sunglasses to wear in the sun. While it's true that Justin Beiber wears Ray Bans, this does not necessarily mean that they are the best sunglasses to wear in the sun. There could be many factors that contribute to why a particular pair of sunglasses is the best to wear in the sun, such as price, quality, or style. It's important to consider a variety of factors when evaluating the quality of a product, rather than relying on celebrity endorsements alone.\n###\nanalyze the argument \"{}\"\n",
    "goals": "express the goal of the argument \"The two courses I took at UWI were not very interesting. I don't think its a good university.\"\nIt's possible that the goal of the sentence \"The two courses I took at UWI were not very interesting. I don't think its a good university.\" is to express the speaker's personal feelings and opinions about their academic experience at The University of the West Indies (UWI). The sentence could be communicating the speaker's dissatisfaction with UWI to someone else, or it may simply be a way for the speaker to process their own thoughts and feelings.\n###\nexpress the goal of the argument \"People either like coffee or hate it\"\nIt's possible that the goal of the sentence \"People either like coffee or hate it\" is to make a general observation about the way that people tend to feel about coffee. The sentence suggests that there are two main categories of people when it comes to coffee: those who like it, and those who hate it. The speaker may be trying to make a point about the strong opinions that people tend to have about coffee, or they may be simply making a casual observation.\n###\nexpress the goal of the argument \"Did the pollution you caused increase or decrease your profits?\" without saying that it is not possible without more context.\nIt's possible that the goal of the argument \"Did the pollution you caused increase or decrease your profits?\" is to determine the relationship between pollution and profits. The speaker may be trying to hold the person or entity being addressed accountable for their actions, or they may be trying to gather information about the impact of pollution on profits. The argument could also be intended to raise awareness about the potential consequences of causing pollution.\n###\nexpress the goal of the argument \"Justin Beiber wears Ray Bans, so you should buy a pair to wear in the sun.\" without saying that it is not possible without more context. \nIt's possible that the goal of the argument \"Justin Beiber wears Ray Bans, so you should buy a pair to wear in the sun.\" is to convince the listener to buy a pair of Ray Ban sunglasses. The speaker may be trying to persuade the listener to buy a pair of Ray Bans by appealing to their desire to be like Justin Beiber, or they may be trying to convince the listener that Ray Bans are a good brand of sunglasses.\n###\nexpress the goal of the argument \"The accident was caused by the taxi parking in the street\" without saying that it is not possible without more context. \nIt's possible that the goal of the argument \"The accident was caused by the taxi parking in the street\" is to determine the cause of the accident. The speaker may be trying to hold the taxi driver accountable for the accident, or they may be trying to gather information about the cause of the accident. The argument could also be intended to raise awareness about the potential consequences of parking in the street.\n###\nexpress the goal of the argument \"{}\" without saying that it is not possible without more context.\n",
    "counter": "represent the counter argument to the argument \"four out of five dentists agree that brushing your teeth makes your life meaningful\"\nThere is no evidence to support the claim that brushing your teeth makes your life meaningful. In fact, there are many other factors that contribute to a meaningful life, and it's not fair to suggest that brushing your teeth is a necessary or important factor.\n###\nrepresent the counter argument to the argument \"Chick-Fil-A is the greatest food chain in existence. Look at the crowds!\"\nPopularity does not necessarily equate to quality. There are many factors that contribute to the success of a food chain, such as location, price, marketing efforts, and menu offerings. Just because Chick-Fil-A has large crowds does not necessarily mean that it is the best food chain.\n###\nrepresent the counter argument to the argument \"If you are not with us, you are against us.\"\nThere are often multiple perspectives on an issue, and it's not fair to assume that someone is automatically opposed to a particular viewpoint just because they are not actively supporting it. It's possible to have a nuanced or balanced view that doesn't align with either side completely.\n###\nrepresent the counter argument to the argument \"The accident was caused by the taxi parking in the street\"\nThere are many factors that contribute to an accident, and it's not fair to suggest that the taxi parking in the street was the only factor. It's possible that the accident was caused by a combination of factors, such as the taxi parking in the street, the driver being distracted, and the driver being tired.\n###\nrepresent the counter argument to the argument \"Justin Beiber wears Ray Bans, so you should buy a pair to wear in the sun.\"\nOne counter argument could be that just because Justin Bieber wears Ray Bans, it does not necessarily mean they are the best or most suitable sunglasses for the individual making the decision to purchase them. Personal preferences, face shape, and intended use should also be taken into consideration when choosing sunglasses. Additionally, there are many other brands of sunglasses available that may offer better quality or value for the price.\n###\nrepresent the counter argument to the argument \"{}\"\n",
    "codex_prediction": "classes = ['fallacy of logic', 'circular reasoning', 'appeal to emotion',\n       'intentional', 'faulty generalization', 'fallacy of extension',\n       'false dilemma', 'ad populum', 'ad hominem', 'false causality',\n       'equivocation', 'fallacy of relevance', 'fallacy of credibility']\n-------\n\"Politicians and diapers must be changed often, and for the same reason.\"\nfallacy of logic\n###\nMr. Casal was very tired because he had no energy.\ncircular reasoning\n###\n“All those opposed to my arguments for the opening of a new department, signify by saying, ‘I quit.’\"\nappeal to emotion\n###\nDid the pollution you caused increase or decrease your profits?\nintentional\n###\n10 of the last 14 National Spelling Bee Champions have been Indian American. Indian Americans must all be great spellers!\nfaulty generalization\n###\nIf you don't drive a car, you hate the Earth.\nfallacy of extension\n###\nPeople either like coffee or hate it.\nfalse dilemma\n###\nJustin Beiber wears Ray Bans, so you should buy a pair to wear in the sun.\nad populum\n###\nHe was born to Catholic parents and raised as a Catholic until his confirmation in 8th grade.  Therefore, he is bound to want to defend some Catholic traditions and, therefore, cannot be taken seriously.\nad hominem\n###\nThe accident was caused by the taxi parking in the street\nfalse causality\n###\nAll living beings come from other living beings.  Therefore, the first forms of life must have come from a living being.  That living being is God.\nequivocation\n###\nService Tech: Your car could use some new tires. Bart: You have a financial interest in selling me tires, why should I trust you? Service Tech: You brought your car to me to have it checked, sir. Bart: I brought my car to the shop where you work. Service Tech: So should we forget about the new tires for now? Bart: I never suggested that.  Are you trying to use reverse psychology on me, so I will buy the tires?\nfallacy of relevance\n###\n“four out of five dentists agree that brushing your teeth makes your life meaningful”\nfallacy of credibility\n###\n\"George Bush is a good communicator because he speaks effectively.\"\ncircular reasoning\n###\n\"ALL teenagers are irresponsible\"\nfaulty generalization\n###\n\"{}\"",
}


def query_openai(prompt, text, api_key):

    openai.api_key = api_key

    response = openai.Completion.create(
        model="code-davinci-002",
        prompt=prompt.format(text),
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["###"]
    )
    return response.choices[0].text


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, required=True, help="feature to extract", choices=[
        'counter',
        'codex_prediction',
        'structure',
        'explanations',
        'goals'
    ])
    parser.add_argument("--api_key", type=int, required=True,
                        default=0, help="OpenAI API key to use")
    parser.add_argument("--replace", action="store_true", default=False,
                        help="Whether to replace existing extracted features")
    parser.add_argument("--split", type=str,
                        help="which split of the data to use", choices=['train', 'dev', 'test', 'climate_test'])

    args = parser.parse_args()

    if args.split is not None:
        splits = [args.split]
    else:
        splits = ["dev", "test", "train", "climate_test"]

    print(
        f"starting crawling with feature {args.feature} and api_key {args.api_key}")

    for split in splits:

        data_df = pd.read_csv(f"{data_dir}/{split}.csv")

        if args.feature in data_df.columns and not args.replace:
            print(f'{args.feature} Already done for split {split}')
            continue

        data_texts = data_df["text"].tolist()
        pbar = tqdm(total=len(data_texts), leave=False)
        data_features = []

        cnt = 0
        while True:
            try:
                if cnt == len(data_texts):
                    break
                extracted_feature = query_openai(
                    prompt=prompts_dict[args.feature],
                    text=data_texts[cnt],
                    api_key=api_key_to_use[args.api_key]
                )
                print(f"{cnt} {data_texts[cnt]}")
                print(f"{cnt} {extracted_feature}")
                print("=====================================")
                cnt += 1
                pbar.update(1)
                data_features.append(extracted_feature)

            except Exception as e:
                print(e)
                time.sleep(30)
                continue

        joblib.dump(data_features,
                    f"{data_dir}/{split}_{args.feature}.pkl")
        data_df[args.feature] = data_features
        data_df.to_csv(f"{data_dir}/{split}.csv", index=False)
