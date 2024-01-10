import openai
import time
import csv
import pandas as pd


prompt_prefix = """
You are a data generator now.
Please imagine yourself as a human user who is using the AI video editing tool and you only need to enter text commands to control the video clips.
Please follow the examples below and use your imagination to generate a few queries. The data should be natural and random.
The type of data you need to generate now is "{}", the following are some examples.
{}
{}
{}
{}
{}
{}
{}
{}
Now, please generate one more queries:
"""


def generate_query_batch_data():
    data = pd.read_csv('data/frameIo_dataset_cleansed.csv')
    labels = set(data['Label'].unique())

    result = []

    for label in labels:
        label_data = data.loc[data['Label'] == label]

        sample_data = label_data.sample(n=8)
        sample_list = sample_data.iloc[:, 0].tolist()

        prompt = prompt_prefix.format(label, *sample_list)

        res = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0.1)["choices"][0]["text"].strip()      
        result.append(res)
        time.sleep(1)

    return result


if __name__ == "__main__":

    batch = generate_query_batch_data()
    print(batch)
