import fire
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
)
from neo_stif.components.trying import PointingConverter
from neo_stif.components.utils import (
    create_label_map,
    get_pointer_and_label,
    process_masked_source,
)
import pandas as pd
from neo_stif.components.train_data_preparation import prepare_data_tagging_and_pointer
import datasets
from neo_stif.train_insertion import insertion
from neo_stif.train_tagger_pointer import taggerpoint
import rich
from neo_stif.infer import generate_felix
import time


console = rich.console.Console()


MAX_MASK = 30
USE_POINTING = True
model_dict = {"koto": "indolem/indobert-base-uncased"}
LR_TAGGER = 5e-5  # due to the pre-trained nature
LR_POINTER = 1e-5  # no pre-trained
LR_INSERTION = 2e-5  # due to the pre-trained nature
VAL_CHECK_INTERVAL = 20

prompt = """
Given an informal or colloquial sentences of Bahasa Indonesia, translate the informal sentence to its standard or formal one. 
Give the answer with this format:
<<informal>> : <<formal>>
For example

{string_few_shot}

Just give the answer directly! Do not ELABORATE and GIVE INFORMATION! I am poor.

<<{instance}>> : 
"""


def open_ai_inference(
    dev_csv_path="data/stif_indo/test_with_pointing.csv",
    test_csv_path="data/stif_indo/test_with_pointing.csv",
    output_path="data/pred/stif_chat_gpt.txt",
    sleep_time=2,
):
    """
    Perform felix inference task using the given parameters.
    """
    from openai import OpenAI

    client = OpenAI()
    dev_csv = pd.read_csv(dev_csv_path)
    test_csv = pd.read_csv(test_csv_path)

    def generate_example(informal, formal):
        return f"<<{informal}>> : <<{formal}>>"

    string_few_shot = ""
    for i in range(3):
        few_shot_1 = generate_example(dev_csv.iloc[i].informal, dev_csv.iloc[i].formal)
        string_few_shot += few_shot_1 + "\n"
    result = []

    for i in tqdm(range(len(test_csv))):
        prompt_ready = prompt.format(
            string_few_shot=string_few_shot, instance=test_csv.iloc[i].informal
        )
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt_ready},
            ],
        )

        # save it to a file append 'chat.txt'
        with open(output_path, "a") as f:
            f.write(str(i) + "\n")
            f.write(completion.choices[0].message.content + "\n")
            f.write("\n")
            f.write("====\n")

        result.append(completion.choices[0].message.content)
        time.sleep(sleep_time)


def generate_data_for_felix_insertion(
    data_path,
    out_path,
    src: str = "informal",
    tgt: str = "formal",
    tokenizer_name: str = "indolem/indobert-base-uncased",
):
    label_dict = create_label_map(MAX_MASK, USE_POINTING)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    df = pd.read_csv(data_path)
    data = datasets.Dataset.from_pandas(df)
    data, label_dict = prepare_data_tagging_and_pointer(data, tokenizer, label_dict)
    console.log("Loaded data from [red]{}[/red]".format(data_path))
    console.log("Creating data for insertion...")
    data_processed = data.map(
        process_masked_source,
        batched=False,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label_map": label_dict,
            "src": src,
            "tgt": tgt,
        },
        num_proc=12,
    )
    console.log("Created data for insertion")
    console.log("Saving to [red]{}[/red]".format(out_path))
    data_processed.save_to_disk(out_path)


def generate_data_for_felix_tagging(
    data_path,
    out_path,
    src: str = "informal",
    tgt: str = "formal",
    tokenizer_name: str = "indolem/indobert-base-uncased",
):
    """
    Generate data for Felix.

    Args:
        data_path (str): The path to the input data file.
        out_path (str): The path to save the generated data.
        src (str, optional): The column name for the source text. Defaults to "informal".
        tgt (str, optional): The column name for the target text. Defaults to "formal".
        tokenizer_name (str, optional): The name of the tokenizer to use. Defaults to "indolem/indobert-base-uncased".
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    df = pd.read_csv(data_path)
    console.print("Loaded data from {}".format(data_path))
    # replace such words with xWORDx
    df[src] = df[src].str.replace("xxx", "x")
    df[tgt] = df[tgt].str.replace("xxx", "x")

    label_dict = create_label_map(MAX_MASK, USE_POINTING)
    point_converter = PointingConverter({}, False)
    console.print("Created label map and pointing converter")
    a = df.apply(
        lambda x: get_pointer_and_label(
            x,
            label_dict,
            point_converter,
            tokenizer,
            src=src,
            tgt=tgt,
        ),
        axis=1,
    )
    # unpack to two series
    point_indexes, label = zip(*a)
    df["point_indexes"] = point_indexes
    df["label"] = label

    df.to_csv(out_path, index=False)
    console.print("Saved to {}".format(out_path))


def train_stif(
    part: str = "taggerpointer",
    model="koto",
    batch_size=32,
    with_validation=False,
    do_compute_class_weight=False,
    device="cuda",
    train_path="data/stif_indo/train_with_pointing.csv",
    dev_path="data/stif_indo/dev_with_pointing.csv",
    processed_train_data_path="data/stif_indo/train_insertion",
    processed_dev_data_path="data/stif_indo/dev_insertion",
    output_dir_path="output/stif-i-f/felix-tagger-pointer/",
    from_scratch=False,
):
    """
    Trains the STIF model.

    Args:
        part (str): The part of the model to train. Options are "taggerpointer" and "insertion".
        model (str): The name of the model to use.
        batch_size (int): The batch size for training.
        with_validation (bool): Whether to use a validation dataset during training.
        do_compute_class_weight (bool): Whether to compute class weights for imbalanced datasets.
        device (str): The device to use for training. Defaults to "cuda".
        train_path (str): The path to the training data.
        dev_path (str): The path to the validation data.
        processed_train_data_path (str): The path to the processed training data.
        processed_dev_data_path (str): The path to the processed validation data.
        output_dir_path (str): The path to the output directory.
        from_scratch (bool): Whether to train the model from scratch.

    Raises:
        ValueError: If an invalid part is specified.

    Returns:
        None
    """

    tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
    label_dict = create_label_map(MAX_MASK, USE_POINTING)

    # Callback for trainer

    df_train = pd.read_csv(train_path)
    data_train = datasets.Dataset.from_pandas(df_train)
    data_train, label_dict = prepare_data_tagging_and_pointer(
        data_train, tokenizer, label_dict
    )
    model_path_or_name = model_dict[model]

    if with_validation:
        df_dev = pd.read_csv(dev_path)

    if part == "taggerpointer":
        taggerpoint(
            df_train,
            data_train,
            tokenizer,
            batch_size,
            model_path_or_name,
            with_validation,
            do_compute_class_weight,
            device,
            label_dict,
            df_dev,
            USE_POINTING,
            output_dir_path,
            LR_TAGGER,
            from_scratch=from_scratch,
        )
    elif part == "insertion":
        insertion(
            processed_train_data_path,
            processed_dev_data_path,
            tokenizer,
            batch_size,
            model_path_or_name,
            device,
            label_dict,
            output_dir_path,
            LR_INSERTION,
            from_scratch=from_scratch,
        )
    else:
        raise ValueError("Invalid part: {}".format(part))


if __name__ == "__main__":
    fire.Fire()
