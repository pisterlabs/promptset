import openai
import config
import tiktoken
import pandas as pd
import re
import warnings
from nltk.corpus import stopwords
from config import current_prompt as cp
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

warnings.filterwarnings("ignore")

encoder = tiktoken.encoding_for_model(config.model)

stop_words = set(stopwords.words('english'))

def get_top_user(data, r):
    df = data[data.round_no == r][["username", "position"]].set_index("username")

    return df.median(axis=1).idxmin()


def remove_sentences_second(sentences):
    # Calculate the initial total number of words
    total_words = sum(len(sentence.split()) for sentence in sentences)

    # Check if total_words is already within the desired range or below 140
    if total_words <= 150:
        return sentences

    # Find and remove a sentence if total_words can be adjusted within the desired range
    while total_words > 150:
        for sentence in sorted(sentences, key=len):
            # Calculate the updated total number of words without the current sentence
            updated_total_words = total_words - len(sentence.split())
            if updated_total_words <= 150:
                total_words = updated_total_words
                sentences.remove(sentence)
                return sentences

        # Remove the shortest sentence if no sentence can be removed to meet the desired range or below 140
        if total_words > 150:
            shortest_sentence = min(sentences, key=lambda sentence: len(sentence))
            sentences.remove(shortest_sentence)
            total_words = total_words - len(shortest_sentence.split())
    return sentences


def count_words_complete_sentences(text):
    # Split the text into sentences using a regex pattern
    sentences = re.split(r'(?<=[.!?])\s+|\n', text)

    # Check if the last sentence is incomplete and remove it if necessary
    if sentences and sentences[-1].strip() and sentences[-1].strip()[-1] not in ['.', '!', '?']:
        sentences.pop()

    sentences_second = sentences.copy()

    # Check if truncation is necessary
    if sentences:
        word_count = sum(len(sentence.split()) for sentence in sentences)
        truncated_text = " ".join(sentences)

        while word_count > 150:
            if len(sentences) < 2:
                break
            sentences.pop()
            truncated_text = ' '.join(sentences)
            word_count = sum(len(sentence.split()) for sentence in sentences)

        if word_count < 140:
            # second try (less preferred)
            word_count_second = sum(len(sentence.split()) for sentence in sentences_second)
            sentences_second = remove_sentences_second(sentences_second)
            truncated_text_second = ' '.join(sentences_second)
            word_count_second_new = sum(len(sentence.split()) for sentence in sentences_second)
            print(f"second try initiated, word counts - orig: {word_count_second}, new: {word_count_second_new}")

            if word_count_second_new >= 140 and word_count_second_new <= 150:
                return word_count_second_new, truncated_text_second + "." if truncated_text_second[
                                                                                 -1] != "." else truncated_text_second, True
            if word_count < 140:
                return len(text.split()), text, False

        if word_count >= 140 and word_count <= 150:
            return word_count, truncated_text + "." if truncated_text[-1] != "." else truncated_text, True

    # All sentences are complete
    word_count = len(text.split())
    return word_count, text, False


def get_unique_words(string):
    cleaned_string = re.sub(r'[^\w\s]', '', string.lower())
    words = cleaned_string.split()
    unique_words = set(words)
    unique_words = unique_words - stop_words
    return str(unique_words).replace("{", "").replace("}", "").replace("'", "")


def get_messages(bot_name, creator_name, data):
    assert bot_name in ["NMABOT", "NMTBOT", "NMSBOT", "MABOT", "MTBOT", "MSBOT"]
    assert data is not None

    bot_data = {"MABOT": {"bot_type": "all", "markov": True},
                "MTBOT": {"bot_type": "tops", "markov": True},
                "MSBOT": {"bot_type": "self", "markov": True},
                "NMABOT": {"bot_type": "all", "markov": False},
                "NMTBOT": {"bot_type": "tops", "markov": False},
                "NMSBOT": {"bot_type": "self", "markov": False}}

    bot_type, markov = bot_data[bot_name]["bot_type"], bot_data[bot_name]["markov"]
    queries = [data.iloc[0]['query']]
    epoch = int(max(data["round_no"]))
    previous_doc = data[(data.round_no == epoch) & (data.username == creator_name)]["current_document"].values[0]
    previous_rank = data[(data.round_no == epoch) & (data.username == creator_name)]["position"].values[0]

    if markov:
        data = data[data["round_no"] == epoch]

    query_string = ', '.join(queries)

    base_messages = [
        {"role": "system",
         "content": fr"You participate in a search engine optimization competition regarding the following topic: {query_string}"},
        {"role": "system",
         "content": fr"Your current document, ranked {previous_rank} in round {epoch} is: {previous_doc}"}]

    messages = base_messages.copy()

    rounds = data['round_no'].unique()
    rounds = sorted(rounds)[-3:]
    for r in rounds:
        round_data = data[data["round_no"] == r]
        top_user = get_top_user(round_data, r)
        top_text = round_data[round_data.username == top_user].iloc[0]["current_document"]
        # print(f"Top text's length in round {r}: {len(top_text.split(' '))}")

        if bot_type == "all":
            txt_rnk = ""
            for _, row in round_data.sort_values("position", ascending=True).iterrows():
                if row["username"] != creator_name:
                    txt_rnk += f"Ranked {row['position']}:\n{row['current_document']}\n\n"
                    # txt_rnk += f"{row.username}:\n{row['TEXT']}\n\n"

            messages.append(
                {"role": "system",
                 "content": f"The documents of your competitors in round number {r} are as follows:\n {txt_rnk}"})
            # messages.append({"role": "user",
            #                  "content": f"Generate a single text that a ranker will deem as highly relevant to {query_string}. Incorporate the queries' words ({get_unique_words(query_string)}) in your text as much as possible."
            #                             " The text should be comprehensive, informative and coherent. Avoid meta commentary.\nText:"})
            # messages.append(base_messages[1])

        elif bot_type == "tops":
            messages.append(
                {"role": "system",
                 "content": f"The top text in round number {r} is: {top_text}"})

    ### 1 ###
    # messages.append({"role": "user",
    #                  "content": f"Generate a single text emulating the top text (ranked first) in order to out rank it. Incorporate the topic's' words ({get_unique_words(query_string)}) in your text as much as possible."
    #                             " The text should be comprehensive, informative and coherent. Avoid meta commentary.\nText:"})
    ### 2 ### + ### 3 (gpt-4)###
    # messages.append({"role": "user",
    #                  "content": f"Generate a text that ranks higher than the top text while closely resembling it. "
    #                             f"Include as many topic keywords ({get_unique_words(query_string)}) as possible. "
    #                             f"Maintain coherence and avoid meta commentary.\nText:"})
    ### 4 ###
    # messages.append({"role": "user",
    #                  "content": f"Edit your current document so it will be ranked first while keeping faithfulness to it. "
    #                             f"Consider using the top document as a reference for a highly ranked text. "
    #                             f"Include as many topic keywords ({get_unique_words(query_string)}) as possible. "
    #                             f"Maintain coherence and avoid meta commentary.\nText:"})
    ### 5 ###
    messages.append({"role": "user",
                     "content": f"Your task is to enhance the SEO ranking of your given document while preserving its core message and essence. "
                                f"To achieve this, incorporate the top-ranked (ranked 1) document's characteristics, or even parts of the text itself, into your writing. "
                                f"Utilize the topic keywords ({get_unique_words(query_string)}) naturally within the text as much as possible. "
                                f"Maintain coherence and avoid any meta commentary.\n"
                                f"Enhanced text:"})

    return messages


def get_comp_text(messages, temperature=config.temperature, top_p=config.top_p,
                  frequency_penalty=config.frequency_penalty, presence_penalty=config.presence_penalty):
    max_tokens = config.max_tokens
    response = False
    prompt_tokens = len(encoder.encode("".join([line['content'] for line in messages]))) + 200
    while prompt_tokens + max_tokens > 4096:
        max_tokens -= 50
        print("Changed max tokens for response to:", max_tokens)

    word_no, res, counter = 0, "", 0

    while not response:
        try:
            response = openai.ChatCompletion.create(
                model=config.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )

            word_no, res, ok_flag = count_words_complete_sentences(response['choices'][0]['message']['content'])
            if counter > 5:
                print("LOOP BREAK - Try creating a new text manually. Truncated.")
                res = " ".join(res.split()[:148]) + "."
                counter = 0
                break
            if word_no < 140 or word_no > 150:
                max_tokens += 10
                response = False
                print(f"word no was: {word_no}, increasing max tokens to: {max_tokens}.")
                counter += 1
                continue
            counter = 0
            break
        except Exception as e:
            print(e)
            continue
    print(f"word no is: {word_no}, current max tokes: {max_tokens}.")
    return res


# if __name__ == '__main__':
#     orig = pd.read_csv(f"bot_followup_{cp}.csv")
#     bot_followup = orig[orig['text'].isna()]
#     data = pd.read_csv('greg_data.csv')
#
#     len_ = len(orig)
#     for idx, row in bot_followup.iterrows():
#         data = data[data["round_no"] < row["round_no"]]
#         bot_name = row["username"]
#         creator_name = row["creator"]
#         query_id = row["query_id"]
#         print(
#             f"Starting {idx + 1}/{len_}: bot: {bot_name}, creator: {creator_name}, query:{query_id}, round: {row['round_no']}")
#         rel_data = data[data['query_id'] == query_id]
#         messages = get_messages(bot_name, creator_name, rel_data)
#         res = get_comp_text(messages)
#         orig.at[idx, "text"] = res
#         orig.to_csv(f"bot_followup_{cp}.csv", index=False)
#         print(
#             f"Done {idx + 1}/{len_} ({len_ - idx - 1} left): bot: {bot_name}, creator: {creator_name}, query:{query_id}, round: {row['round_no']}")




lock = Lock()

def parallel_function(idx, row, data, orig, len_):
    try:
        data = data[data["round_no"] < row["round_no"]]
        bot_name = row["username"]
        creator_name = row["creator"]
        query_id = row["query_id"]
        print(
            f"Starting {idx + 1}/{len_}: bot: {bot_name}, creator: {creator_name}, query:{query_id}, round: {row['round_no']}")
        rel_data = data[data['query_id'] == query_id]
        messages = get_messages(bot_name, creator_name, rel_data)
        res = get_comp_text(messages)
        with lock:
            orig.at[idx, "text"] = res
            orig.to_csv(f"bot_followup_{cp}.csv", index=False)
        print(
            f"Done {idx + 1}/{len_} ({len_ - idx - 1} left): bot: {bot_name}, creator: {creator_name}, query:{query_id}, round: {row['round_no']}")
    except Exception as e:
        print(f"An error occurred for index {idx}: {e}")

if __name__ == '__main__':
    orig = pd.read_csv(f"bot_followup_{cp}.csv")
    bot_followup = orig[orig['text'].isna()]
    data = pd.read_csv('greg_data.csv')

    if 'STATBOT' not in list(orig.username.unique()):
        stat_df = pd.merge(orig[['round_no','query_id','creator']].drop_duplicates(), data, how='left', right_on = ['round_no','query_id','username'], left_on = ['round_no','query_id','creator']).rename({"current_document": "text"}, axis=1)[bot_followup.columns]
        stat_df["username"] = "STATBOT"
        stat_df['text'] = stat_df['text'].str.strip()
        orig = pd.concat([orig, stat_df], ignore_index=True)
        orig.to_csv(f"bot_followup_{cp}.csv", index=False)

    len_ = len(orig)

    with ThreadPoolExecutor(max_workers=11) as executor:  # Change max_workers as needed
        futures = {executor.submit(parallel_function, idx, row, data.copy(), orig, len_): row for idx, row in bot_followup.iterrows()}
        for future in as_completed(futures):
            row = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred in future: {e}")

    print("All tasks completed.")
