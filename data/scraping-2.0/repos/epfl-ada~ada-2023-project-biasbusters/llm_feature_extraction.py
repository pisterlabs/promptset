import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import time

full_df = pd.read_csv("./ada-2023-project-biasbusters/backup.csv")

def get_query_complex(summaries_batch, keywords_dict):

    assert len(summaries_batch) == 5, "need to be 5 texts in a batch"

    intro = (
        "Analyze the following five movie plot summaries:\n\n"
        f"First movie plot summary:\n{summaries_batch[0]}\n"
        f"Second movie plot summary:\n{summaries_batch[1]}\n"
        f"Third movie plot summary:\n{summaries_batch[2]}\n"
        f"Fourth movie plot summary:\n{summaries_batch[3]}\n"
        f"Fifth movie plot summary:\n{summaries_batch[4]}\n"
    )

    question = (
        "\nYou have five questions:\n"

        "1. Identify the relevant themes present. "
        "Use the following fixed list of themes keywords for your analysis: ") + \
        ", ".join(keywords_dict["identify_topic"]) + (
        ".\nList only those relevant themes that are present in this list. If a theme is not present, "
        "do not include it in your response. Your response should be a comma-separated list of single "
        "keywords, reflecting the themes found in the plot summary, with no additional information.\n"

        "2. Identify the overall mood of the plot. "
        "Use the following fixed list of themes keywords for your analysis: ") + \
        ", ".join(keywords_dict["identify_mood"]) + (
        ".\nList only those relevant moods that are present in this list. If a mood is not present, "
        "do not include it in your response. Your response should be a comma-separated list of single "
        "keywords, reflecting the moods found in the plot summary, with no additional information.\n"

        "3. Identify the target audience of the plot. "
        "Use the following fixed list of target audience types for your analysis: ") + \
        ", ".join(keywords_dict["identify_target_audience"]) + (
        ".\nProvide only one the most relevant target audience type that is present in this list. "
        "Do not provide a target audience type that is not present in the list. "
        "Your response should be just a word from the list with no additional information.\n"

        "4. Identify the historical period of the plot. "
        "Use the following fixed list of historical period types for your analysis: ") + \
        ", ".join(keywords_dict["identify_temporal_setting"]) + (
        ".\nProvide only one the most relevant historical period type that is present in this list. "
        "Do not provide a historical period type that is not present in the list. "
        "Your response should be just a word from the list with no additional information.\n"

        "5. Identify the primary geographical setting of the plot. "
        "Use the following fixed list of geographical setting types for your analysis: ") + \
        ", ".join(keywords_dict["identify_location_setting"]) + (
        ".\nProvide only one the most relevant geographical setting type that is present in this list. "
        "Do not provide a geographical settingd type that is not present in the list. "
        "Your response should be just a word from the list with no additional information.\n"

        "\nAnswer these five questions for every movie plot summary from above in the following format:"
        "\nFirst movie plot summary:"
        "\n1.#YOURANSWER\n2.#YOURANSWER\n3.#YOURANSWER\n4.#YOURANSWER\n5.#YOURANSWER"
        "\nSecond movie plot summary:"
        "\n1.#YOURANSWER\n2.#YOURANSWER\n3.#YOURANSWER\n4.#YOURANSWER\n5.#YOURANSWER"
        "\nThird movie plot summary:"
        "\n1.#YOURANSWER\n2.#YOURANSWER\n3.#YOURANSWER\n4.#YOURANSWER\n5.#YOURANSWER"
        "\nFourth movie plot summary:"
        "\n1.#YOURANSWER\n2.#YOURANSWER\n3.#YOURANSWER\n4.#YOURANSWER\n5.#YOURANSWER"
        "\nFifth movie plot summary:"
        "\n1.#YOURANSWER\n2.#YOURANSWER\n3.#YOURANSWER\n4.#YOURANSWER\n5.#YOURANSWER"
        
    )
    return intro + question

keywords_dict = {"identify_topic": ["Romance", "Conflict", "Adventure", "Mystery", "Comedy",
                                    "Tragedy", "Fantasy", "Science Fiction", "Horror", "Drama",
                                    "Action", "Historical"],

                "identify_mood": ["Exciting", "Dark", "Lighthearted", "Romantic",
                                "Inspirational", "Dramatic", "Fantastical"],

                "identify_target_audience": ["Children", "Teenagers", "Adults", "Families", "Elderly"],

                "identify_temporal_setting": ["past", "modern", "future"],

                "identify_location_setting": ["real", "fictional"]}


def parse_llm_output(llm_output, num_texts = 5, num_features = 5):
    # Split the input into individual movie summaries
    movie_summaries = llm_output.strip().split("\n\n")

    if len(movie_summaries) != num_texts:
        raise ValueError(f"Invalid format: Must be {num_texts} texts in a request.")

    # Initialize a 5x5 array to store the parsed data
    parsed_data = []

    # Process each movie summary
    for summary in movie_summaries:
        # Split the summary into its components
        components = summary.split("\n")[1:]  # skip the first line as it's a header

        # Check if the format is correct (5 components per movie)
        if len(components) != num_features:
            raise ValueError("Invalid format: Each movie summary must have 5 components.")

        # Process each component
        cleaned_components = []
        for component in components:
            # Remove the number and split the component into words
            words = component.split('. ', 1)[1] if '. ' in component else component
            words_list = words.split(', ') if ', ' in words else [words]
            cleaned_components.append(np.array(words_list))

        # Add the cleaned components to the parsed data array
        parsed_data.append(cleaned_components)

    return np.array(parsed_data, dtype='object')

from openai import OpenAI

client = OpenAI(
    api_key = "YOUR_OPENAI_API_KEY",
)


llm_features_df = pd.DataFrame(index=np.arange(full_df.shape[0]), 
                               columns=['topic', 'mood', 'target_audience', 'temporal_setting', 'location_setting'])

start_idx = 0

query_size = 5
num_requests = 200

end_idx = start_idx + num_requests * query_size

start_end_idxs = np.arange(start_idx, end_idx + 1, query_size)

for i, (batch_start_idx, batch_end_idx) in tqdm(enumerate(zip(start_end_idxs[:-1], start_end_idxs[1:]))):

    start_time = time.time() # to meet a request per minute limit

    batch = full_df.iloc[batch_start_idx : batch_end_idx]["plot_summary"].values
    
    response_status = False
    num_attempts = 0
    while not response_status:
        try:
            response = client.chat.completions.create(
                    messages=[
                    {"role": "system", "content": "You're a famous movie critic. You are very knowledgeable about movie plots, their genres, their specifics and their emotional aspects. Your answers are precise and concise. You can't be wrong in your answers, as it would ruin your reputation."},
                    {"role": "user", "content": get_query_complex(summaries_batch = batch, keywords_dict = keywords_dict)}
                ],
                    model="gpt-3.5-turbo-1106",
                )

            
            parsed_output = parse_llm_output(response.choices[0].message.content)
            response_status = True
        except ValueError as e:
            num_attempts += 1

            if num_attempts > 5:
                print(f"idxs from {batch_start_idx} to {batch_end_idx} are not processed")
                response_status = True
                parsed_output = np.nan
            
            end_time = time.time()  # Record the end time of the iteration
            iteration_duration = end_time - start_time  # Calculate the duration of this iteration

            if iteration_duration < 20.05:
                time.sleep(20 - iteration_duration)  # Pause for the remaining time

            start_time = time.time()

    if parsed_output is not np.nan:
        llm_features_df.iloc[batch_start_idx : batch_end_idx] = parsed_output.squeeze()
    else:
        llm_features_df.iloc[batch_start_idx : batch_end_idx] = parsed_output

    end_time = time.time()  # Record the end time of the iteration
    iteration_duration = end_time - start_time  # Calculate the duration of this iteration

    if iteration_duration < 20:
        time.sleep(20 - iteration_duration)  # Pause for the remaining time

llm_features_df.to_csv("llm_features_extracted.csv", index=False)