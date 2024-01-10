import time
from multiprocessing import Pool
import configparser
import openai
import pandas as pd

NUM_PROCESSES = 10

config = configparser.ConfigParser()
config.read('config.ini')

openai.api_type = config.get('openai', 'api_type')
openai.api_base = config.get('openai', 'api_base')
openai.api_version = config.get('openai', 'api_version')
openai.api_key = config.get('openai', 'api_key')
deployment_name = config.get('deployment', 'name')

prompt_messages_history = [
    {"role": "system",
     "content": "You are an AI assistant. You are provided with examples of medical conversations between doctor and patient. Insert conversational filters like \"um\", \"uh\" and \"mm-hmm\" as you see fit in each patient turn to make the conversation more casual."},
]


def get_generated_conversation(convo, prompt_messages):
    """
    Method to generate the conversation for a given medical note
    :param convo: str
    :param prompt_messages: list[dict]
    :return: (str, str)
    """
    final_prompt_messages = prompt_messages + \
                            [{"role": "user", "content": "[INPUT CONVERSATION]" + convo}]

    try:
        response = openai.ChatCompletion.create(
            temperature=0.75,
            deployment_id=deployment_name,
            messages=final_prompt_messages
        )
        generated_conversation = response["choices"][0]["message"]["content"]

        if "[doctor]" not in generated_conversation or "[patient]" not in generated_conversation:
            return convo, ""

        generated_conversation = clean_artifacts_from_generated_conversation(generated_conversation)
        return convo, generated_conversation

    except:
        return convo, ""


def clean_artifacts_from_generated_conversation(convo):
    """
    Clean the generated conversation by only retaining content from the first [doctor] tag
    :param convo: str
    :return: str
    """
    start_index = convo.lower().index("[doctor]")
    cleaned_response = convo[start_index:].lower()
    return cleaned_response


if __name__ == "__main__":
    notes_dataset = pd.read_csv("sample_generated_data.csv")
    prompt_notes = notes_dataset["Dialogue"].values.tolist()
    score = notes_dataset["Score"].values.tolist()

    score_dict = {k: v for k, v in zip(prompt_notes, score)}
    arguments = list(zip(prompt_notes, [prompt_messages_history] * len(prompt_notes)))

    start_time = time.time()
    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.starmap(get_generated_conversation, arguments)
        scores = [score_dict[x[0]] for x in results]
        results_combined = [[x[0], x[1], y] for x, y in zip(results, scores) if x[1] != ""]

        dataframe_data = pd.DataFrame(results_combined)
        dataframe_data.columns = ["Original Dialogue", "Dialogue with Fillers", "Score"]
        dataframe_data.to_csv("sample_generated_data_with_fillers.csv", index=False)
    end_time = time.time()

    print(end_time - start_time)
