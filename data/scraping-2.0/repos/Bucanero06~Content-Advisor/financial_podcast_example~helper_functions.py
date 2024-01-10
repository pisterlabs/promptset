import numpy as np


def preprocess_questions_naive(raw_file_name, output_file_name=None):
    """Data Manipulation for questions, for now specific to current sample"""
    if output_file_name is not None: assert isinstance(output_file_name, str)
    import csv
    import pandas as pd
    df = pd.DataFrame(columns=['episode', 'url', 'start_timestamp', 'start', 'end', 'question', 'context'])
    i = 0
    with open(raw_file_name) as csv_file:
        reader = csv.reader(csv_file)

        # skip the header row in the csv file
        next(reader)

        for row in reader:
            # assign each column in the row to a variable and split questions on carriage return
            episode, url, questions = row
            question_list = questions.split("\n")

            # for each question in the list, extract the timestamp and convert it to seconds for youtube
            for question in question_list:
                pieces = question.split('-')
                timestamp = pieces[0]
                minutes, seconds = timestamp.split(':')
                seconds = int(seconds) + (int(minutes.lstrip()) * 60)

                # add a new row to the dataframe
                df.loc[i] = [episode, url, timestamp, seconds, seconds, " ".join(pieces[1:]), ""]

                try:
                    df.loc[i - 1]['end'] = df.loc[i]['start']
                except:
                    print(f"skipping row {i} because there is no previous row")

                i += 1

                df['end'][df['end'] < df['start']] = 0
                df['end'][df['start'] == df['end']] = 0
    if output_file_name:
        df.to_csv(output_file_name)

    return df


def is_part_of_question(segment, start, end):
    if segment['start'] > start:
        if segment['end'] < end or end == 0:
            return True

    return False


def combine_episodes(input_dir, prefix, output_file):
    # import pandas as pd
    # df = pd.DataFrame()
    # import os
    # episodes_numbers_list = list(map(lambda x: x.split('_')[-1].split('.')[0], os.listdir(input_dir)))
    # for episode in episodes_numbers_list:
    #     episode_df = pd.read_csv(f'{input_dir}/{prefix}_{episode}.csv')
    #     df = df.append(episode_df)
    # df.to_csv(output_file)

    # optimize the above code
    import pandas as pd
    import glob
    import os

    pwd = os.getcwd()
    os.chdir(input_dir)
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])

    print(f'Saving combined csv to {output_file}, includes episodes files: {all_filenames}')
    os.chdir(pwd)
    combined_csv.to_csv(output_file, index=False, encoding='utf-8-sig')
    return combined_csv


# def get_question_context(row, transcription_output):
def get_question_context(row):
    global transcription_output

    question_segments = list(
        filter(lambda segment: is_part_of_question(segment, row['start'], row['end']),
               transcription_output['segments']))
    # include question from timestamp in the context
    context = row['question']
    for segment in question_segments:
        context += segment['text']

    return context


def ask_question(episode_df, pre_context_prompt, question, top_n_context=4,
                 completion_model="text-davinci-003",
                 embedding_model='text-embedding-ada-002',
                 temperature=1,
                 max_tokens=500,
                 top_p=1,
                 frequency_penalty=0,
                 presence_penalty=0,
                 ):
    from openai.embeddings_utils import get_embedding, cosine_similarity

    question_vector = get_embedding(question, engine=embedding_model)

    print(f'{question_vector = }')
    print(f'{question = }')
    print(f'{episode_df = }')

    # episode_df["similarities"] = episode_df['embedding'].apply(lambda x: cosine_similarity(x, question_vector))
    # np.array(eval(a)), np.array(b) included -------------------------------------------------------vvvvv bug waiting to happen
    episode_df["similarities"] = episode_df['embedding'].apply(
        # lambda x: cosine_similarity(np.array(
        #     eval(x) if isinstance(x, str) or isinstance(x, bytes)
        #     else x if not None else 0,
        #     np.array(question_vector)
        # )
        # )
        #
        lambda x: cosine_similarity(np.array(eval(x)), np.array(question_vector)

                                    ))

    episode_df = episode_df.sort_values("similarities", ascending=False).head(top_n_context)

    print(f'{episode_df = }')
    #
    # episode_df.to_csv("sorted.csv")

    context = []
    for i, row in episode_df.iterrows():
        context.append(row['context'])

    context = "\n".join(context)

    # prompt = f"""Answer the following question using only the context below. Answer in the style of Ben Carlson a financial advisor and podcaster. If you don't know the answer for certain, say I don't know.
    #
    # Context:
    # {context}
    #
    # Q: {question}
    # A:"""

    # prompt = f"""{pre_context_prompt}
    #
    # Context:
    # {context}
    #
    # Q: {question}
    # A:"""

    prompt = f"""{pre_context_prompt} Q: {question} A:"""
    print(f'{prompt = }')

    import openai
    completion = openai.Completion.create(
        prompt=prompt,
        engine=completion_model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )["choices"][0]["text"].strip(" \n")

    return completion
