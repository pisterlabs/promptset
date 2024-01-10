import pandas as pd
import openai
import os
from dotenv import load_dotenv


def paraphrase_sentence(sentence):
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    initial_prompt = "You are an assistant that paraphrases sentences"
    example_sentence = "What is a planet?"
    example_paraphrase = "What is the definition of a planet?"

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": initial_prompt},
            {"role": "user", "content": "Paraphrase the following sentence:{}".format(example_sentence)},
            {"role": "assistant", "content": example_paraphrase},
            {"role": "user", "content": "Paraphrase the following sentence:{}".format(sentence)}
        ]
    )

    paraphrasis = completion.choices[0].message.content.strip()

    return paraphrasis


def augment_df(df):
    # Create an empty dataframe to store augmented data
    augmented_df = pd.DataFrame(columns=['question', 'answer', 'tag'])

    # Iterate over the rows of the original dataframe
    for index, row in df.iterrows():
        # Paraphrase the question and answer
        paraphrased_question = paraphrase_sentence(row['question'])
        paraphrased_answer = paraphrase_sentence(row['answer'])
        tag = row['tag']

        # Add the original question and answer to the augmented dataframe
        augmented_df = pd.concat([augmented_df, pd.DataFrame(
            {'question': [row['question']], 'answer': [row['answer']], 'tag': [tag]})], ignore_index=True)
        
        # Add the paraphrased question and answer to the augmented dataframe
        augmented_df = pd.concat([augmented_df, pd.DataFrame(
            {'question': [paraphrased_question], 'answer': [paraphrased_answer], 'tag': [tag]})], ignore_index=True)

    return augmented_df
