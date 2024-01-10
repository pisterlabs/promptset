from tqdm import tqdm
import csv
import openai

openai.api_key = "OPENAI_API_KEY"


def paraphrase(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        messages=[
            {"role": "system", "content": "Paraphrase the sentence."},
            {"role": "user", "content": text},
        ]
    )
    openai_response = response['choices'][0]['message']['content'].strip()
    return openai_response


def augment_data(arguments, amount=2):
    augmented_arguments = {}
    for arg in tqdm(arguments, desc="Augmenting data"):
        id, conclusion, stance, premise = arg.strip().split("\t")
        augmented_arguments[id] = [conclusion, stance, premise]
        for i in range(amount):
            augmented_arguments[f"{id}-{i}"] = [conclusion, stance, paraphrase(premise)]
    return augmented_arguments


def store_augmented_data(augmented_arguments: dict):
    with open("./datasets/arguments-training-small-augmented.tsv", "w") as file:
        writer = csv.writer(file, delimiter="\t")
        for argument_id, argument_data in augmented_arguments.items():
            writer.writerow([argument_id] + argument_data)


if __name__ == '__main__':
    # open the arguments file and the labels-training file and augment the data
    with open("./datasets/arguments-training-small.tsv", "r") as arguments_file:
        next(arguments_file)  # Skip the header line
        arguments = arguments_file.readlines()
        augmented_arguments = augment_data(arguments)
        print(augmented_arguments)
        store_augmented_data(augmented_arguments)
