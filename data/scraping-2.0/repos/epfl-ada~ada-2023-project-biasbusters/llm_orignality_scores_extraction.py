from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import re

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

full_df = pd.read_csv("backup_with_llm_features.csv")
selected_summaries_lengths = full_df['plot_summary'].apply(lambda x: len(x))
long_summaries_df = full_df[selected_summaries_lengths >= 530]

def get_query_complex(summaries_batch):

    assert len(summaries_batch) == 5, "need to be 5 texts in a batch"

    intro = (
        "Analyze the following five movie plot summaries that are missing last 2 sentences:\n\n"
        f"First movie plot summary:\n{summaries_batch[0]}\n\n"
        f"Second movie plot summary:\n{summaries_batch[1]}\n\n"
        f"Third movie plot summary:\n{summaries_batch[2]}\n\n"
        f"Fourth movie plot summary:\n{summaries_batch[3]}\n\n"
        f"Fifth movie plot summary:\n{summaries_batch[4]}\n"
    )

    question = (
        "\nYou need to complete these five movie plot summaries with two sentences that end the corresponding stories. "
        "Try to guess the endings of the stories as accurately as possible based on the fragments provided and your own knowledge of the movie plot lines."

        "\nWrite four different most possible options of endings (two sentences for each options) for each of the five movies above in the following format:"
        "\n\nFirst movie plot ending options:"
        "\n1. #YOURANSWER\n2. #YOURANSWER\n3. #YOURANSWER\n4. #YOURANSWER"
        "\n\nSecond movie plot ending options:"
        "\n1. #YOURANSWER\n2. #YOURANSWER\n3. #YOURANSWER\n4. #YOURANSWER"
        "\n\nThird movie plot ending options:"
        "\n1. #YOURANSWER\n2. #YOURANSWER\n3. #YOURANSWER\n4. #YOURANSWER"
        "\n\nFourth movie plot ending options:"
        "\n1. #YOURANSWER\n2. #YOURANSWER\n3. #YOURANSWER\n4. #YOURANSWER"
        "\n\nFifth movie plot ending options:"
        "\n1. #YOURANSWER\n2. #YOURANSWER\n3. #YOURANSWER\n4. #YOURANSWER"

    )
    return intro + question

def parse_llm_output(llm_output, num_texts = 5, num_options = 4):
    # Split the input into individual movie summaries
    movie_endings = llm_output.strip().split("\n\n")

    if len(movie_endings) != num_texts:
        raise ValueError(f"Invalid format: Must be {num_texts} texts in a request.")

    # Initialize a 5x5 array to store the parsed data
    parsed_data = []

    # Define valid first lines
    valid_first_lines = ["First movie plot ending options:",
                        "Second movie plot ending options:",
                        "Third movie plot ending options:",
                        "Fourth movie plot ending options:",
                        "Fifth movie plot ending options:"]

    # Process each movie summary
    for ending in movie_endings:

        # Split the text into lines
        lines = ending.strip().split('\n')
        if len(lines) != num_options + 1:
            raise ValueError(f"Invalid format: Must be {num_options} options in an output.")

        # Check if the first line is valid
        if lines[0].strip() not in valid_first_lines:
            raise ValueError("Invalid first line of a block. Must be one of the specified options.")

        # Initialize an empty list to store answers
        answers = []

        # Regular expression pattern to match lines with index and actual answer
        pattern = re.compile(r'^\d\.\s.+$')

        # Process each line except the first
        for line in lines[1:]:
            # Check if the line matches the required format
            if pattern.match(line):
                # Extract the answer and add to the list
                answer = line.split('. ', 1)[1]  # Split on the first occurrence of '. ' and take the second part
                answers.append(answer)
            else:
                raise ValueError(f"Invalid line format: {line}")

        # Convert the answers list to a numpy array
        parsed_data.append(np.array(answers))

    return np.array(parsed_data, dtype='object')

from openai import OpenAI

client = OpenAI(
    api_key = 'YOUR_OPENAI_API_KEY',
)

llm_ending_predictions = pd.DataFrame(index = np.arange(long_summaries_df.shape[0]),
                                      columns = [*[f"llm_ending_prediction_{i}" for i in range(4)],
                                                 *[f"Similarity_score_{i}" for i in range(4)],
                                                 "Avarage_similarity_score"])


start_idx = 0

query_size = 5
num_requests = 200

end_idx = start_idx + num_requests * query_size

start_end_idxs = np.arange(start_idx, end_idx + 1, query_size)

for i, (batch_start_idx, batch_end_idx) in tqdm(enumerate(zip(start_end_idxs[:-1], start_end_idxs[1:]))):

    start_time = time.time() # to meet a request per minute limit

    batch = long_summaries_df.iloc[batch_start_idx : batch_end_idx]["plot_summary"]
    truncated_batch = batch.apply(lambda x: '.'.join(x.split('.')[:-3])).values # truncating last 3 because last element is always empty
    endings_batch = batch.apply(lambda x: '.'.join(x.split('.')[-3:])).values

    response_status = False
    num_attempts = 0
    while not response_status:
        try:
            response = client.chat.completions.create(
                    messages=[
                    {"role": "system", "content": "You're a famous movie critic. You are very knowledgeable about movie plots, their genres, their specifics and their emotional aspects. Your answers are precise and concise. You can't be wrong in your answers, as it would ruin your reputation."},
                    {"role": "user", "content": get_query_complex(summaries_batch = truncated_batch)}
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

        llm_ending_predictions.iloc[batch_start_idx : batch_end_idx, : 4] = parsed_output

        for block_idx in range(parsed_output.shape[0]):

            emb_gt= model.encode(endings_batch[block_idx], normalize_embeddings=True)
            mean_score = 0

            for option_idx in range(parsed_output.shape[1]):
                emb_predected = model.encode(parsed_output[block_idx][option_idx], normalize_embeddings=True)
                score = emb_predected @ emb_gt

                llm_ending_predictions[f"Similarity_score_{option_idx}"].iloc[batch_start_idx + block_idx] = score
                mean_score += score
            mean_score /= parsed_output.shape[1]

            llm_ending_predictions["Avarage_similarity_score"].iloc[batch_start_idx + block_idx] = mean_score
    else:
        llm_ending_predictions.iloc[batch_start_idx : batch_end_idx] = parsed_output

    end_time = time.time()  # Record the end time of the iteration
    iteration_duration = end_time - start_time  # Calculate the duration of this iteration

    if iteration_duration < 20:
        time.sleep(20.5 - iteration_duration)  # Pause for the remaining time

llm_ending_predictions.to_csv("llm_originality_scores.csv", index=False)