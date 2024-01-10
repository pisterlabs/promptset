# !pip install openai

import os
import time
import numpy as np
import json
from openai import OpenAI
import random
import argparse
import hashlib
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--prompt", type=str, default="simple", help="")
parser.add_argument("--length_limit", type=int, default=8, help="")
parser.add_argument("--num_cand", type=int, default=19, help="")
parser.add_argument("--random_seed", type=int, default=2023, help="")
parser.add_argument("--api_key", type=str, default="sk-", help="")
parser.add_argument("--use_cache", type=bool, default=False, help="")
parser.add_argument("--create_cache", type=bool, default=False, help="")
parser.add_argument("--verbose", type=bool, default=False, help="")

args = parser.parse_args()

prompt_options = ["simple", "chain_of_thought", "features"]
print(f"prompt: {args.prompt}")
if args.prompt not in prompt_options:
    raise ValueError("prompt must be either 'simple', 'chain_of_thought' or 'features'")

rseed = args.random_seed
random.seed(rseed)


def read_json(file):
    with open(file) as f:
        return json.load(f)


def write_json(data, file):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


data_ml_100k = read_json("./ml_100k.json")

# print (data_ml_100k[0][0])
# print (data_ml_100k[0][1])
# print (len(data_ml_100k))

client = OpenAI(api_key=args.api_key)

u_item_dict = {}
u_item_p = 0
for elem in data_ml_100k:
    seq_list = elem[0].split(" | ")
    for movie in seq_list:
        if movie not in u_item_dict:
            u_item_dict[movie] = u_item_p
            u_item_p += 1
print(len(u_item_dict))
u_item_len = len(u_item_dict)

user_list = []
for i, elem in enumerate(data_ml_100k):
    item_hot_list = [0 for ii in range(u_item_len)]
    seq_list = elem[0].split(" | ")
    for movie in seq_list:
        item_pos = u_item_dict[movie]
        item_hot_list[item_pos] = 1
    user_list.append(item_hot_list)
user_matrix = np.array(user_list)
user_matrix_sim = np.dot(user_matrix, user_matrix.transpose())


pop_dict = {}
for elem in data_ml_100k:
    # elem = data_ml_100k[i]
    seq_list = elem[0].split(" | ")
    for movie in seq_list:
        if movie not in pop_dict:
            pop_dict[movie] = 0
        pop_dict[movie] += 1


i_item_dict = {}
i_item_id_list = []
i_item_user_dict = {}
i_item_p = 0
for i, elem in enumerate(data_ml_100k):
    seq_list = elem[0].split(" | ")
    for movie in seq_list:
        if movie not in i_item_user_dict:
            item_hot_list = [0.0 for ii in range(len(data_ml_100k))]
            i_item_user_dict[movie] = item_hot_list
            i_item_dict[movie] = i_item_p
            i_item_id_list.append(movie)
            i_item_p += 1
        #         item_pos = item_dict[movie]
        i_item_user_dict[movie][i] += 1
#     user_list.append(item_hot_list)
i_item_s_list = []
for item in i_item_id_list:
    i_item_s_list.append(i_item_user_dict[item])
#     print (sum(item_user_dict[item]))
item_matrix = np.array(i_item_s_list)
item_matrix_sim = np.dot(item_matrix, item_matrix.transpose())

id_list = list(range(0, len(data_ml_100k)))


### user filtering
def sort_uf_items(target_seq, us, num_u, num_i):
    candidate_movies_dict = {}
    sorted_us = sorted(list(enumerate(us)), key=lambda x: x[-1], reverse=True)[:num_u]
    dvd = sum([e[-1] for e in sorted_us])
    for us_i, us_v in sorted_us:
        us_w = us_v * 1.0 / dvd
        #         print (us_i)
        us_elem = data_ml_100k[us_i]
        #         print (us_elem[0])
        #         assert 1==0
        us_seq_list = us_elem[0].split(" | ")  # +[us_elem[1]]

        for us_m in us_seq_list:
            #             print (f"{us_m} not in {target_seq}, {us_m not in target_seq}")
            #             break

            if us_m not in target_seq:
                if us_m not in candidate_movies_dict:
                    candidate_movies_dict[us_m] = 0.0
                candidate_movies_dict[us_m] += us_w

    #         assert 1==0

    candidate_pairs = list(
        sorted(candidate_movies_dict.items(), key=lambda x: x[-1], reverse=True)
    )
    #     print (candidate_pairs)
    candidate_items = [e[0] for e in candidate_pairs][:num_i]
    return candidate_items


### item filtering
def soft_if_items(target_seq, num_i, total_i, item_matrix_sim, item_dict):
    candidate_movies_dict = {}
    for movie in target_seq:
        #         print('ttt:',movie)
        sorted_is = sorted(
            list(enumerate(item_matrix_sim[item_dict[movie]])),
            key=lambda x: x[-1],
            reverse=True,
        )[:num_i]
        for is_i, is_v in sorted_is:
            s_item = i_item_id_list[is_i]

            if s_item not in target_seq:
                if s_item not in candidate_movies_dict:
                    candidate_movies_dict[s_item] = 0.0
                candidate_movies_dict[s_item] += is_v
    #             print (item_id_list[is_i], candidate_movies_dict)
    candidate_pairs = list(
        sorted(candidate_movies_dict.items(), key=lambda x: x[-1], reverse=True)
    )
    #     print (candidate_pairs)
    candidate_items = [e[0] for e in candidate_pairs][:total_i]
    #     print (candidate_items)
    return candidate_items


"""
In order to economize, our initial step is to identify user sequences that exhibit a high probability of obtaining accurate predictions from GPT-3.5 based on their respective candidates. 
Subsequently, we proceed to utilize the GPT-3.5 API to generate predictions for these promising user sequences.
"""
results_data_15 = []
length_limit = args.length_limit
num_u = 12
total_i = args.num_cand
count = 0
total = 0
cand_ids = []
for i in id_list[:1000]:
    elem = data_ml_100k[i]
    seq_list = elem[0].split(" | ")

    candidate_items = sort_uf_items(
        seq_list, user_matrix_sim[i], num_u=num_u, num_i=total_i
    )

    #     print (elem[-1], '-',seq_list[-1])

    if elem[-1] in candidate_items:
        #         print ('HIT: 1')
        count += 1
        cand_ids.append(i)
    else:
        pass
    #         print ('HIT: 0')
    total += 1
print(f"count/total:{count}/{total}={count*1.0/total}")
print("-----------------\n")

model = "gpt-3.5-turbo-instruct"

# if .cache directory does not exist create it
if (args.use_cache or args.create_cache) and not os.path.exists("./.cache"):
    os.makedirs("./.cache")


def get_response(input):
    hashed_name = hashlib.md5((model + "-" + input).encode("utf-8")).hexdigest()
    cache_file = f"./.cache/{hashed_name}"

    if args.use_cache and os.path.exists(cache_file):
        if args.verbose:
            print(f"Using cache for {input}")
        with open(cache_file) as f:
            response_text = f.read()
    else:
        try_nums = 5
        kk_flag = 1
        while try_nums:
            try:
                response = client.completions.create(
                    model=model,
                    prompt=input,
                    max_tokens=512,
                    temperature=0,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    n=1,
                )
                try_nums = 0
                kk_flag = 1
            except Exception as e:
                time.sleep(1)
                try_nums = try_nums - 1
                kk_flag = 0

        if kk_flag == 0:
            time.sleep(5)
            response = client.completions.create(
                model=model,
                prompt=input,
                max_tokens=256,
                temperature=0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                n=1,
            )
        response_text = response.choices[0].text
        if args.create_cache:
            if args.verbose:
                print(f"Creating cache for {input}")
            with open(cache_file, "w") as f:
                f.write(response_text)

    return response_text


def get_chain_prompt(candidate_items, seq_list):
    prompt = """
        Do not print results for step 1 or 2. Think in your head. Only print the results of step 3.
        Candidate Set (candidate movies): {}.
        The movies I have watched (watched movies): {}.
        Step 1: Think about what features may be most important to me when selecting movies.
        Step 2: Select the most featured movies (at most 5 movies) from the watched movies according to my preferences in descending order (Format: [no. a watched movie.]).
        Step 3: Recommend 10 movies from the Candidate Set similar to the selected movies I've watched (Format: [no. a watched movie - a candidate movie]).
    """.format(
        ", ".join(candidate_items),
        ", ".join(seq_list),
    )

    return prompt


# current best: 0.54 HR@10
def get_simple_prompt(candidate_items, seq_list):
    prompt = """
        Candidate Set (candidate movies): {}.
        The movies I have watched (watched movies): {}.
        Recommend 10 movies from the Candidate Set similar to the movies I've watched (Format: [no. a watched movie - a candidate movie]).
    """.format(
        ", ".join(candidate_items),
        ", ".join(seq_list),
    )

    return prompt


if args.prompt == "features":
    # Define the column names based on the description
    column_names = [
        "movie_id",
        "movie_title",
        "release_date",
        "video_release_date",
        "IMDb_URL",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    # Read the data from the file into a DataFrame
    movie_df = pd.read_csv(
        "./ml-100k/u.item", sep="|", names=column_names, encoding="latin-1"
    )

    # Remove the last 7 characters from the movie_title column
    movie_df["movie_title"] = movie_df["movie_title"].str[:-7]

    # Extract the last 4 characters from the release_date column
    movie_df["release_date"] = movie_df["release_date"].str[-4:]

    # Create an empty dictionary to store movie titles and genres
    movie_genre_dict = {}
    # Create an empty dictionary to store movie titles and release years
    movie_year_dict = {}

    # Iterate through rows in the DataFrame
    for index, row in movie_df.iterrows():
        # Extract movie title, genres and release year
        movie_title = row["movie_title"]
        genres = [
            genre
            for genre, value in row.items()
            if value == 1 and genre != "movie_title" and genre != "movie_id"
        ]
        release_year = (
            int(row["release_date"]) if pd.notna(row["release_date"]) else "Unknown"
        )  # Convert release_date to integer

        # Add entry to the dictionary
        movie_genre_dict[movie_title] = genres
        movie_year_dict[movie_title] = release_year


def get_prompt_features(candidate_items, seq_list, movie_year_dict, movie_genre_dict):
    # Format candidate movies with release year and genres
    formatted_candidate_movies = [
        "{} ({}, {})".format(
            movie,
            movie_year_dict.get(movie),
            ", ".join(movie_genre_dict.get(movie, [])),
        )
        for movie in candidate_items
    ]

    # Format watched movies with release year and genres
    formatted_watched_movies = [
        "{} ({}, {})".format(
            movie,
            movie_year_dict.get(movie),
            ", ".join(movie_genre_dict.get(movie, [])),
        )
        for movie in seq_list
    ]

    prompt = """
        Candidate Set (candidate movies with release year and genres): {}.
        The movies I have watched (watched movies with release year and genres): {}.
        Recommend 10 movies from the Candidate Set similar to the movies I've watched (Format: [no. a watched movie - a candidate movie]).
    """.format(
        ", ".join(formatted_candidate_movies),
        ", ".join(formatted_watched_movies),
    )

    return prompt


count = 0
total = 0
results_data = []
for i in cand_ids[:]:  # [:10] + cand_ids[49:57] + cand_ids[75:81]:
    elem = data_ml_100k[i]
    seq_list = elem[0].split(" | ")[::-1]

    candidate_items = sort_uf_items(
        seq_list, user_matrix_sim[i], num_u=num_u, num_i=total_i
    )

    # choosing not to shuffle for now as that breaks the cache
    # random.shuffle(candidate_items)

    if args.prompt == "features":
        input = get_prompt_features(
            candidate_items, seq_list[-length_limit:], movie_year_dict, movie_genre_dict
        )
    elif args.prompt == "chain_of_thought":
        input = get_chain_prompt(candidate_items, seq_list[-length_limit:])
    else:
        input = get_simple_prompt(candidate_items, seq_list[-length_limit:])

    predictions = get_response(input)

    hit_ = 0
    if elem[1] in predictions:
        count += 1
        hit_ = 1
    else:
        pass
    total += 1

    if args.verbose:
        print(f"GT:{elem[1]}")
        print(f"predictions:{predictions}")

    # print (f"GT:{elem[-1]}")
    print(f"PID:{i}; count/total:{count}/{total}={count*1.0/total}\n")
    result_json = {
        "PID": i,
        "Input": input,
        "GT": elem[1],
        "Predictions": predictions,
        "Hit": hit_,
        "Count": count,
        "Current_total": total,
        "Hit@10": count * 1.0 / total,
    }
    results_data.append(result_json)

file_dir = (
    f"./results_multi_prompting_len{length_limit}_numcand_{total_i}_seed{rseed}.json"
)
write_json(results_data, file_dir)
