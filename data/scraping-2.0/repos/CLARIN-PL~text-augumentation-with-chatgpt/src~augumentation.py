import openai
from dataset.persent import PerSenTDataset
from dataset.multiemo import MultiEmoDataset
from os.path import join, isfile
import pandas as pd
from tqdm import tqdm
import time

def save_df(i, dictionary, filename):
    df = pd.DataFrame(data=dictionary)
    if i == 0:
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode="a", index=False, header=False)

def get_response(messages):
    not_done = True
    too_long = False
    curr_response = ""
    while not_done:
        try:
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )
            curr_response = chat.choices[0].message.content
            not_done = False
        except openai.error.InvalidRequestError:
            too_long = True
            not_done = False
        except (openai.error.ServiceUnavailableError, openai.error.APIError):
            time.sleep(5)
    curr_response = curr_response.replace("\n", " ")
    return curr_response, too_long

def prepare_first_message(template, x, y):
    form = {"text": x}
    if "{sentiment}" in template:
        form["sentiment"] = y
    curr_content = template.format(**form)
    message = [{"role": "user", "content": curr_content}]
    return message

def prepare_second_message(template, msg, first_response):
    curr_content = template
    msg.append({"role": "assistant", "content": first_response})
    msg.append({"role": "user", "content": curr_content})
    return msg


def create_one_augumentation(dataset, template, filename, threshold=10):
    augumented_data = {"DOCUMENT": [], "TRUE_SENTIMENT": []}
    # messeges = [{"role": "user", "content": "{content}"}]
    if isfile(filename):
        saved = pd.read_csv(filename)
        saved_examples = saved.shape[0]
    else:
        saved_examples = 0
    for i, (x, y) in tqdm(list(enumerate(zip(dataset.X, dataset.Y)))):
        if i < saved_examples:
            continue
        curr_msg = prepare_first_message(template, x, y)
        curr_response, too_long = get_response(curr_msg)
        if too_long:
            continue
        augumented_data["DOCUMENT"].append(curr_response)
        augumented_data["TRUE_SENTIMENT"].append(y)
        if i % threshold == 0:
            save_df(i, augumented_data, filename)
            augumented_data = {"DOCUMENT": [], "TRUE_SENTIMENT": []}
    if len(augumented_data["DOCUMENT"]) > 0:
        save_df(-1, augumented_data, filename)
    return filename

def prepare_collections(filename, suffix):
    final_filename = filename.replace(".csv", f"_{suffix}.csv")
    data = {"DOCUMENT": [], "TRUE_SENTIMENT": []}
    if isfile(final_filename):
        saved = pd.read_csv(final_filename)
        saved_examples = saved.shape[0]
    else:
        saved_examples = 0
    return final_filename, data, saved_examples

def create_many_augumentations(dataset, templates, filename, threshold=10):
    first_filename, first_data, first_saved = prepare_collections(filename, "normal")
    second_filename, second_data, second_saved = prepare_collections(filename, "different_words")
    assert first_saved == second_saved

    for i, (x, y) in tqdm(list(enumerate(zip(dataset.X, dataset.Y)))):
        if i < first_saved:
            continue
        first_msg = prepare_first_message(templates[0], x, y)
        first_response, too_long = get_response(first_msg)
        if too_long:
            continue
        first_data["DOCUMENT"].append(first_response)
        first_data["TRUE_SENTIMENT"].append(y)

        second_msg = prepare_second_message(templates[1], first_msg, first_response)
        second_response, _ = get_response(second_msg)

        second_data["DOCUMENT"].append(second_response)
        second_data["TRUE_SENTIMENT"].append(y)

        if i % threshold == 0:
            save_df(i, first_data, first_filename)
            save_df(i, second_data, second_filename)
            first_data = {"DOCUMENT": [], "TRUE_SENTIMENT": []}
            second_data = {"DOCUMENT": [], "TRUE_SENTIMENT": []}
    if len(first_data["DOCUMENT"]) > 0:
        save_df(-1, first_data, first_filename)
    if len(second_data["DOCUMENT"]) > 0:
        save_df(-1, second_data, second_filename)
    return first_filename, second_filename


def persent_augumentation(new_file_name, template):
    datadir = "../data/PerSenT"
    train_filepath = "train.csv"
    dataset = PerSenTDataset(join(datadir,train_filepath))
    
    filename = join(datadir, new_file_name)
    if not isinstance(template, str):
        create_many_augumentations(dataset, template, filename, threshold=10)
    else:
        create_one_augumentation(dataset, template, filename, threshold=10)
        
def multiemo_augumentation(new_file_name, template):
    datadir = "../data/multiemo2"
    train_filepath = "all.text.train.en.txt"
    dataset = MultiEmoDataset(join(datadir,train_filepath))
    
    filename = join(datadir, new_file_name)
    if not isinstance(template, str):
        create_many_augumentations(dataset, template, filename, threshold=10)
    else:
        create_one_augumentation(dataset, template, filename, threshold=10)
    
def templates(type_):
    templates = {"one_paraphrase": ("Generate a paraphrase for the following text, preserving the sentiment of the following statement: {text}", 
                       "Generate another paraphrase by changing more words also keeping the sentiment"),    
                #  "three_paraphrases": "Generate 3 different paraphrases for the following text, preserving the sentiment of the following statement: {text}",
                 "new_text":"Based on the given text, generate another text with a completely new theme, but be inspired by the original text and keep the sentiment of the old one in the new text. Original text: {text}",
                 "new_text_v2":"Based on the given text, generate another text with a completely new theme, but be inspired by the original text and keep a {sentiment} sentiment. Original text: {text}" }
    return templates[type_]

def main():
    openai.api_key = "" 
    template = templates("new_text_v2")
    filename = "all.text.new_train_v2.en.csv"
    # persent_augumentation(filename, template)
    multiemo_augumentation(filename, template)
    

if __name__ == "__main__":
    main()