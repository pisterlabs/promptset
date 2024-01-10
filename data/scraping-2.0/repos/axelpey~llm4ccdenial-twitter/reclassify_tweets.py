import re
from typing import List, Tuple
import openai
import pandas as pd
import json
from datetime import datetime

from constants import (
    RELEVANT_TIMESPANS,
    make_hydrated_df_name,
    make_relevant_df_name,
    CLASSIFIER_VERSION,
)

openai.api_key = open("openai_key.txt").read()


def prompt_gpt35_turbo(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return completion["choices"][0]["message"]["content"]


def make_cc_stance_classification_prompt(tweets: List[Tuple[int, str]]):
    if CLASSIFIER_VERSION == 1:
        return make_cc_stance_classification_prompt_v3(tweets)
    elif CLASSIFIER_VERSION == 2:
        return make_cc_stance_classification_prompt_v2(tweets)


def make_cc_stance_classification_prompt_v1(tweets: List[Tuple[int, str]]):
    tweets = [f'({tweet_id},"{tweet_text}")' for tweet_id, tweet_text, _ in tweets]
    prompt = "I want you to classify these tweets as coming from believer or denier."
    prompt += "For each tweet, give me:\n"
    prompt += " - The tweet id\n"
    prompt += " - A number between 0 and 1: 1 if the author likely to be a climate change believer, 0 if it's likely to be a denier, 0.5 if you're unsure."
    prompt += " - A reason for your classification, in maximum 10 words."
    prompt += "Give me the results as a list of lists, with no line jumps, like this:\n"
    prompt += "[[tweet_id, stance, reason], [tweet_id, stance, reason], ...]"
    prompt += 'tweet_id is an integer, stance is either 0, 0.5 or 1, reason is a string surrounded with the "" quotes.'
    prompt += "If you cannot classify a tweet, give it a stance of -1 and give the reason why."
    prompt += "\n\n"
    prompt += "\n".join(tweets)
    return prompt


def make_cc_stance_classification_prompt_v3(tweets: List[Tuple[int, str]]):
    tweets = [f'({tweet_id},"{tweet_text}")' for tweet_id, tweet_text, _ in tweets]
    prompt = "I want you to classify these tweets as coming from believer or denier, and give me a sentiment and aggressivity score.\n"
    prompt += "For each tweet, give me:\n"
    prompt += " - The tweet id\n"
    prompt += " - A number between 0 and 1: Precisely 1 if the author likely to be a climate change believer, 0 if it's likely to be a denier, 0.5 if you're unsure.\n"
    prompt += " - A float between 0 and 1: closer to 1 if the tweet is positive, closer to 0 if it's negative, closer to 0.5 if it's neutral.\n"
    prompt += " - A number between 0 and 1: Precisely 1 if the tweet is aggressive, 0 if it's not aggressive.\n"
    prompt += "Give me the results for each tweet on a new line, like this:\n"
    prompt += "tweet_id, stance, sentiment, aggressivity\n"
    prompt += "tweet_id is an integer, stance is either 0, 0.5 or 1, sentiment is floating between 0 and 1, aggressivity is either 0 or 1.\n"
    prompt += "If you cannot classify a tweet stance, give it a stance of -1."
    prompt += "\n\n"
    prompt += "\n".join(tweets)
    return prompt


def make_cc_stance_classification_prompt_v2(tweets: List[Tuple[int, str]]):
    tweets = [
        f'({tweet_id},"{tweet_text}", "{tweet_author_description}")'
        for tweet_id, tweet_text, tweet_author_description in tweets
    ]

    prompt = "I want you to classify these tweets as coming from believer or denier."
    prompt += "I will give you a list for which each tweet data will be (tweet id, tweet text, self-written description of the author)\n"
    prompt += "For each tweet, give me:\n"
    prompt += " - The tweet id\n"
    prompt += " - A number between 0 and 1: 1 if the author is likely to be a climate change believer, 0 if it's likely to be a denier, 0.5 if you're unsure."
    prompt += " - A reason for your classification, in maximum 10 words."
    prompt += "Give me the results as a list of lists, with no line jumps, like this:\n"
    prompt += "[[tweet_id, stance, reason], [tweet_id, stance, reason], ...]"
    prompt += 'tweet_id is an integer, stance is either 0, 0.5 or 1, reason is a string surrounded with the "" quotes.'
    prompt += "If you cannot classify a tweet, give it a stance of -1 and give the reason why."
    prompt += "\n\n"
    prompt += "[" + ",".join(tweets) + "]"
    return prompt


def get_text_to_use(df, tweet_id):
    # If the text is an RT, get the original tweet text
    return df[df["id"] == tweet_id]["text"].values[0]
    if df[df["id"] == tweet_id]["text"].values[0].startswith("RT @"):
        ref_text = json.loads(df[df["id"] == tweet_id]["referenced_tweets"].values[0])[
            0
        ]["text"]
        return ref_text
    else:
        return df[df["id"] == tweet_id]["text"].values[0]


def classify_tweets(dates):
    # Get the tweet ids and texts from the hydrated tweets
    df = pd.read_csv(make_hydrated_df_name(dates))
    tweets_ids, tweets_texts = df["id"].values.tolist(), df["text"].values.tolist()
    df["author_description"] = df["author"].apply(
        lambda x: json.loads(x)["description"]
    )
    tweets_author_descriptions = df["author_description"].values.tolist()

    for index, id in enumerate(tweets_ids):
        tweets_texts[index] = get_text_to_use(df, id)

    tweets_ids_texts = list(zip(tweets_ids, tweets_texts, tweets_author_descriptions))
    tweets_ids_texts = tweets_ids_texts[:30]

    res_array = []
    batch_size = 30
    t = datetime.now()

    for i in range(0, len(tweets_ids_texts), batch_size):
        batch = tweets_ids_texts[i : i + batch_size]
        prompt = make_cc_stance_classification_prompt(batch)

        print(f"Sending prompt for batch {i//batch_size + 1} to GPT-3.5-turbo")
        res = prompt_gpt35_turbo(prompt)

        open(
            f"gpt_calls_v{CLASSIFIER_VERSION}/res_{dates}_{i//batch_size + 1}.txt", "w"
        ).write(res)

        try:
            # remove the line jumps at the beginning of res until the first character
            res = re.sub(r"^\s+", "", res)
            res = res.split("\n")
            for l in range(len(res)):
                t_id, stance, sentiment, agg = res[l].split(",")
                res[l] = [int(t_id), float(stance), float(sentiment), int(agg)]

            res_array.extend(
                [
                    {
                        "id": x[0],
                        "stance": x[1],
                        "sentiment": x[2],
                        "aggressivity": x[3],
                        "text": get_text_to_use(df, x[0]),
                    }
                    for x in res
                ]
            )
        except Exception as e:
            print(e)
            print(f"Error with batch {i//batch_size + 1}. Continuing...")
        print(f"Batch {i//batch_size + 1} done in {datetime.now() - t}")
        print(
            f"Estimated time remaining: {(datetime.now() - t) * (len(tweets_ids_texts) - i) / batch_size}"
        )

    df = pd.DataFrame(res_array)

    # Write the results to a file
    df.to_csv(
        f"subsets_v{CLASSIFIER_VERSION}/cc_stance_classification_{dates}.csv",
        index=False,
    )


def compare_classified_tweets():
    # Get the tweet ids classifications from the file generated by classify_tweets
    # Compare the classifications with the original stance in the subset

    for dates in RELEVANT_TIMESPANS[:1]:
        df_relevant = pd.read_csv(make_relevant_df_name(dates))
        df_hydrated = pd.read_csv(make_hydrated_df_name(dates))

        # Get the classifications
        df_relevant["stance"] = df_relevant["stance"].apply(
            lambda x: 1 if x == "believer" else (0.5 if x == "neutral" else 0)
        )
        # Join the two dataframes on the tweet id
        df_original = df_relevant.merge(
            df_hydrated[["id", "text"]], on="id", how="left"
        )[["id", "stance", "sentiment", "aggressiveness"]]
        df_original = df_original.rename(
            columns={
                "stance": "original_stance",
                "sentiment": "original_sentiment",
                "aggressiveness": "original_aggressiveness",
            }
        )

        # Get the classifications from the new classification file
        df_new_classification = pd.read_csv(
            f"subsets_v{CLASSIFIER_VERSION}/cc_stance_classification_{dates}.csv"
        )
        # Merge the two dataframes on the tweet id
        df_merged = df_original.merge(
            df_new_classification, on="id", how="left"
        ).dropna(subset=["stance"])
        df_merged = df_merged.rename(columns={"stance": "new_stance"})

        # Determine the rows where the new and old stances are different
        df_diff = df_merged[df_merged["original_stance"] != df_merged["new_stance"]]

        # Print some stats
        print(f"Total number of tweets: {len(df_merged)}")
        print(f"Number of tweets with different stance: {len(df_diff)}")
        print(
            f"Percentage of tweets with different stance: {len(df_diff)/len(df_merged)*100:.2f}%"
        )

        # Export for manual inspection
        df_diff.to_csv(
            f"manual_inspection_v{CLASSIFIER_VERSION}/diff_cc_stance_classification_2019-03-05_2019-03-25.csv",
        )
        df_merged.to_csv(
            f"manual_inspection_v{CLASSIFIER_VERSION}/merged_cc_stance_classification_2019-03-05_2019-03-25.csv",
        )


if __name__ == "__main__":
    # classify_tweets(RELEVANT_TIMESPANS[0])
    compare_classified_tweets()
