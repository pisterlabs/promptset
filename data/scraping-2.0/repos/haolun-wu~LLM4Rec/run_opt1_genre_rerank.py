import json
from tqdm import tqdm
import openai
from helper.dataloader import map_title_to_id, map_id_to_title
from data.dataloader import get_dataloader
import re
from torch.utils.data import DataLoader, Subset
import torch

access_token = "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"
if torch.cuda.is_available():
    device = torch.device('cuda')
    global_path = '/home/haolun/projects/ctb-lcharlin/haolun/LLM4Rec_User_Summary'
else:
    device = torch.device('cpu')
    global_path = '/Users/haolunwu/Documents/GitHub/LLM4Rec_User_Summary'



def generate_text_openai(prompts, model_name, device):
    MAX_RETRIES = 3
    RETRY_DELAY = 60  # seconds

    if model_name in ['gpt3.5', 'gpt4']:
        if model_name == 'gpt3.5':
            real_model_name = 'gpt-3.5-turbo-16k'
            # real_model_name = 'text-davinci-003'
        elif model_name == 'gpt4':
            real_model_name = 'gpt-4'

        openai.api_key = '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'  # Set your OpenAI API key

        messages_contents = []
        if real_model_name in ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k']:
            for prompt in prompts:
                for attempt in range(MAX_RETRIES):
                    try:
                        response = openai.ChatCompletion.create(
                            model=real_model_name,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant for movie summary."
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            max_tokens=2048  # limit the token length of response
                        )

                        message_content = response['choices'][0]['message']['content']

                        # Remove the assistant's name and the final newline from the message content
                        message_content = message_content.replace("Assistant: ", "").strip()

                        messages_contents.append(message_content)
                        break  # If successful, break out of the retry loop


                    except openai.error.APIError as e:
                        error_message = str(e)
                        if '502 Bad Gateway' in error_message:
                            if attempt < MAX_RETRIES - 1:
                                print(
                                    f"Encountered a 502 error on prompt '{prompt}'. Retrying in {RETRY_DELAY} seconds...")
                                time.sleep(RETRY_DELAY)
                            else:
                                print(f"Failed after {attempt + 1} attempts on prompt '{prompt}'.")
                                raise
                        else:
                            print(f"An unexpected API error occurred: {error_message}")
                            raise


        else:
            # This is for non-chat models like text-davinci-003:
            for prompt in prompts:
                for attempt in range(MAX_RETRIES):
                    try:
                        response = openai.Completion.create(
                            model=real_model_name,
                            prompt=prompt,
                            max_tokens=50  # limit the token length of response
                        )
                        message_content = response['choices'][0]['text'].strip()
                        messages_contents.append(message_content)
                        break  # If successful, break out of the retry loop

                    except openai.error.APIError as e:
                        if e.error['code'] == 502 and attempt < MAX_RETRIES - 1:
                            print(f"Encountered a 502 error on prompt '{prompt}'. Retrying in {RETRY_DELAY} seconds...")
                            time.sleep(RETRY_DELAY)
                        else:
                            print(f"Failed after {attempt + 1} attempts on prompt '{prompt}'.")
                            raise

        return messages_contents


def design_prompt(user_id, genre, movie_id_to_title_mapping, user_summary, watched_movies, base_rankings,
                  rank_movie_num=20):
    preference_summary = user_summary[str(user_id)][genre][8:]
    last_watched_movies = watched_movies[str(user_id)][genre][-10:]
    last_watched_titles_with_ids = [f"{movie['title']} (ID:{movie['movieId']})" for movie in last_watched_movies]
    base_model_ranked_ids = base_rankings[str(user_id)][genre][:rank_movie_num]
    base_model_ranked_titles_with_ids = [f"{movie_id_to_title_mapping[mid]} (ID:{mid})" for mid in
                                         base_model_ranked_ids]

    # Start the prompt by informing the LLM of the purpose
    prompt = "Re-rank the following list of movie titles based on the following information: "
    prompt += "The user preference is:" + preference_summary + " \n"
    # prompt += "Watched movies: " + ", ".join(last_watched_titles_with_ids) + ". \n"
    prompt += f"The list of {rank_movie_num} movie titles (with IDs) from some base model: " + ", ".join(
        base_model_ranked_titles_with_ids) + ". \n"
    prompt += f"Now, please re-rank the above exact {rank_movie_num} movie titles (with IDs) from the base model in the format 'Movie Title (ID:MovieID)'."
    print("prompt:", prompt)
    print("******")

    return prompt


def extract_movie_titles_and_ids_from_response(response_text):
    pattern = re.compile(r'(?P<title>.+?) \(ID:(?P<id>\d+)\)')

    lines = response_text.split('\n')
    movie_lines = [line for line in lines if line.strip() and line.split(" ")[0].replace(".", "").isdigit()]

    extracted_data = [pattern.search(line.split(". ", 1)[1]) for line in movie_lines]
    movie_titles = [match.group('title') for match in extracted_data if match]
    movie_ids = [int(match.group('id')) for match in extracted_data if match]

    return movie_titles, movie_ids


def get_reranked_movie_ids(prompt, movie_title_to_id_mapping):
    response = generate_text_openai([prompt], model_name='gpt3.5', device="cpu")

    reranked_titles, reranked_ids = extract_movie_titles_and_ids_from_response(response[0])
    print("reranked_titles:", reranked_titles)
    print("reranked_ids:", reranked_ids)

    return reranked_ids


# def filter_dataloader(dataloader, target_pairs):
#     indices = []
#     for i, batch in enumerate(dataloader):
#         user_ids = batch["user_id"]
#         genres = batch["genre"]
#
#         for user_id, genre in zip(user_ids, genres):
#             if (str(user_id), genre) in target_pairs:
#                 indices.append(i)
#                 break
#
#     return DataLoader(Subset(dataloader.dataset, indices), batch_size=dataloader.batch_size)


def batch_rerank(global_path, dataloader, movie_id_to_title_mapping, movie_title_to_id_mapping, rank_movie_num=20):
    with open(f'{global_path}/saved_model/ml-100k/BPRMF_user_genre_rankings.json', 'r') as file:
        base_rankings = json.load(file)

    with open(f'{global_path}/saved_user_summary/ml-100k/user_summary_gpt3.5_in1_title0_full.json', 'r') as file:
        user_summary = json.load(file)

    with open(f'{global_path}/data_preprocessed/ml-100k/data_split/train_set_leave_one.json', 'r') as file:
        watched_movies = json.load(file)

    GPT_rankings = {}
    batch_num = 0

    for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
        user_ids = batch["user_id"]
        genres = batch["genre"]

        prompts = []
        for user_id, genre in zip(user_ids, genres):
            prompt = design_prompt(user_id, genre, movie_id_to_title_mapping, user_summary, watched_movies,
                                   base_rankings, rank_movie_num)
            prompts.append((prompt, user_id, genre))

        # Generate responses for the whole batch of prompts at once
        responses = generate_text_openai([p[0] for p in prompts], model_name='gpt3.5', device="cpu")

        # Process responses for each user and genre
        for i, (prompt, user_id, genre) in enumerate(prompts):
            response = responses[i]
            reranked_titles, reranked_ids = extract_movie_titles_and_ids_from_response(response)
            print("reranked_titles:", reranked_titles)
            print("reranked_ids:", reranked_ids)

            if user_id not in GPT_rankings:
                GPT_rankings[user_id] = {}
            GPT_rankings[user_id][genre] = reranked_ids

    # After processing all batches, save the entire GPT_rankings to a single JSON file
    with open(f'{global_path}/saved_model/ml-100k/GPT_BPRMF_user_genre_rankings_all.json', 'w') as file:
        json.dump(GPT_rankings, file)


if __name__ == "__main__":
    # Load the target pairs
    target_pairs_path = f"{global_path}/saved_model/ml-100k/target_user_genre_pairs.json"
    with open(target_pairs_path, 'r') as file:
        target_pairs = set(tuple(pair) for pair in json.load(file))

    print("target_pairs:", len(target_pairs))

    # Load the initial data loader with all the users
    user_genre_file = f"{global_path}/data_preprocessed/ml-100k/user_genre.json"
    user_genre_dataloader = get_dataloader(user_genre_file, batch_size=1, num_users=943, user_start=1, user_end=943,
                                           target_pairs=target_pairs)
    print("user_genre_dataloader:", len(user_genre_dataloader))

    # # Filter the dataloader to only include batches with at least one target pair
    # user_genre_dataloader = filter_dataloader(user_genre_dataloader, target_pairs)
    # print("user_genre_dataloader:", len(user_genre_dataloader))

    # Initialize the movie ID to title and title to ID mappings
    movie_id_to_title_mapping = map_id_to_title("data/ml-100k/movies.dat")
    movie_title_to_id_mapping = map_title_to_id("data/ml-100k/movies.dat")

    # Call the batch rerank function
    batch_rerank(global_path, user_genre_dataloader, movie_id_to_title_mapping, movie_title_to_id_mapping,
                 rank_movie_num=20)
