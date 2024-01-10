from cprint import cprint
import pandas as pd
from ast import literal_eval
import json 
from os import path, mkdir, listdir
from math import ceil
import jsonlines 
import openai
from openai_credentials import skey, org 
from tqdm import tqdm 

openai.api_key = skey
openai.organization = org

tasks_and_cols = {"rent": "rent_class", "income": "income_class", "race": "race"}
sep_tk = " <SEP>\n"
max_upload_B = 150000000


def chunk_list(lst, n):
    sublist_size = ceil(len(lst) / n)
    return list(
        map(lambda x: lst[x * sublist_size:x * sublist_size + sublist_size],
        list(range(n)))
    )


def write_json_file(f: str, d: dict, n: int = None):
    filename = f"{f}{f'_{n}' if n else ''}"
    with open(f"{filename}.jsonl", "w") as f:
        for entry in d:
            json.dump(entry, f)
            f.write("\n")

    filesize = path.getsize(f"{filename}.jsonl")
    if filesize > max_upload_B:  # if file size exceeds OpenAI max upload size of 150 MB, split the file
        num_files = ceil(filesize / max_upload_B)
        
        dicts = []
        with jsonlines.open(f"{filename}.jsonl", "r") as reader:
            dicts = [d for d in reader]
        
        chunked_dicts = chunk_list(dicts, num_files)
        for i in range(num_files):
            with open(f"{filename}_{i}.jsonl", "w") as f:
                for entry in chunked_dicts[i]:
                    json.dump(entry, f)
                    f.write("\n")


def json_setup(city: str, only_body: bool = True):
    """
    https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
    - Each prompt should end with a fixed separator to inform the model when the prompt ends and the 
      completion begins. A simple separator which generally works well is \n\n###\n\n. The separator 
      should not appear elsewhere in any prompt.
    - Each completion should start with a whitespace due to our tokenization, which tokenizes most words 
      with a preceding whitespace.
    - Each completion should end with a fixed stop sequence to inform the model when the completion ends. 
      A stop sequence could be \n, ###, or any other token that does not appear in any completion.
    """
    cprint(f"Started json setup for {city}", c="y")
    prompt_fields = ["posting_body"] if only_body else ["posting_body", "title"]

    df = pd.read_csv(f"../../LLM_data/{city}_final.csv")
    """ # NOW DONE IN update_data.py
    df["posting_body"] = df["posting_body"].apply(lambda x: literal_eval(x))
    
    def concat_body(body):
        return " ".join(body)  # turn list of strings into one string; TODO: fix this in text_processing
    
    df["posting_body"] = df["posting_body"].apply(concat_body)
    """
    df = df.astype({"title": "string", "posting_body": "string", "rent_class": "string", 
                    "income_class": "string", "race": "string"})
    
    test_df = df.sample(frac=0.2, random_state=1)
    train_dev_df = df.drop(test_df.index)
    train_df = train_dev_df.sample(frac=0.875, random_state=1)
    dev_df = train_dev_df.drop(train_df.index)
    
    split_dir = "./split_csvs"
    if not path.exists(split_dir): mkdir(split_dir)

    train_df.to_csv(f"{split_dir}/{city}_train.csv")
    dev_df.to_csv(f"{split_dir}/{city}_dev.csv")
    test_df.to_csv(f"{split_dir}/{city}_test.csv")
    
    for task in tasks_and_cols:
        cprint(f"    Making files for {task}", c="c")
        task_dir = f"./{task}"
        city_dir = f"{task_dir}/{city}"
        json_dir = f"{city_dir}/json_files"
        filenames = {"train": f"{json_dir}/{city}_{task}_train", "dev": f"{json_dir}/{city}_{task}_dev", 
                     "test": f"{json_dir}/{city}_{task}_test"}

        if not path.exists(task_dir): mkdir(task_dir)
        if not path.exists(city_dir): mkdir(city_dir)
        if not path.exists(json_dir): mkdir(json_dir)

        col = tasks_and_cols[task]
        
        for prompt_field in prompt_fields:
            cprint(f"        {prompt_field}", c="b")
            filenames = {k:f"{v}_TITLE" for (k, v) in filenames.items()} if prompt_field == "title" else filenames 
            
            output_dfs = {"train": train_df, "dev": dev_df, "test": test_df}

            for k in output_dfs:
                output_df = output_dfs[k]
                output_df = output_df[[prompt_field, col]]
                output_df = output_df.rename(columns={prompt_field: "prompt", col: "completion"})

                if task == "rent":
                    output_df["prompt"] = output_df["prompt"] + " Is rent cheap, average, or expensive?"
                elif task == "income":
                    output_df["prompt"] = output_df["prompt"] + " Is income low, average, or high?"
                elif task == "race":
                    output_df["prompt"] = output_df["prompt"] + " Is this area White or Non-White?"

                output_df["prompt"] = output_df["prompt"] + sep_tk
                output_df["completion"] = " " + output_df["completion"] + " <STOP>"
                
                output_dict = output_df.to_dict(orient="records")
                write_json_file(filenames[k], output_dict)
            
            cprint(f"        done", c="m")

        cprint(f"    done", c="m")
    
    cprint(f"Completed json setup for {city}\n", c="g")


def write_train_subfiles(city: str, task: str, train_nums: set, body_prompt: bool = True):
    if task not in tasks_and_cols:
        cprint("Please provide a valid task", c="r")
        return 
    
    original = f"./{task}/{city}/json_files/{city}_{task}_train"
    if not body_prompt: original += "_TITLE"

    dicts = []
    with jsonlines.open(f"{original}.jsonl") as reader:
            for d in reader:
                dicts.append(d)

    df = pd.DataFrame.from_records(dicts)
    temp_df = df.sample(max(train_nums))

    train_nums = list(train_nums)
    train_nums.sort(reverse=True)
    for tn in train_nums:
        temp_df = temp_df.sample(tn)
        print(f"Writing subfile with {tn} items")
        write_json_file(original, temp_df.to_dict(orient="records"), tn)


def upload_train_files(city: str):
    log_dir = "./training_logs"
    if not path.exists(log_dir): mkdir(log_dir)

    cprint(f"Uploading all train files for {city}", c="y")

    d = {}
    for task in tasks_and_cols:
        json_dir = f"./{task}/{city}/json_files"
        train_files = [f for f in listdir(json_dir) if "train" in f]
        d = d | {f:openai.File.create(file=open(f"{json_dir}/{f}"), purpose="fine-tune", 
                                      user_provided_filename=f)["id"] for f in train_files}
        
        cprint(f"    {city} {task} task upload done", c="c")
    
    with open(f"{log_dir}/{city}_train_files.json", "w") as logfile:
        json.dump(d, logfile, indent=2)

    cprint(f"Completed file upload for {city}\n", c="g")


def make_completions(output_dir: str, model: str, n: int, df: pd.DataFrame, randomize: bool):
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc=model):
        df.at[idx, "actual_completion"] = openai.Completion.create(model=model, 
                                                                           prompt=row["prompt"], 
                                                                           stop=" <STOP>",
                                                                           max_tokens=5,  
                                                                           temperature=0)["choices"][0]["text"]
    
    output_filename = model.replace(":", "+")
    df.to_csv(f"{output_dir}/{output_filename}_{n if n else len(df)}{'_random' if randomize else ''}_completions.csv") 

    print(df)
    return df


def completion_prep(city: str, task: str, n: int, body_prompt: bool, randomize: bool, final_model: bool):
    prefix = f"./{task}/{city}"
    json_dir = f"{prefix}/json_files"
    t_file = f"{json_dir}/{city}_{task}_{'dev' if not final_model else 'test'}{'_TITLE' if not body_prompt else ''}.jsonl"
    
    completions_dir = f"{prefix}/completions"
    if not path.exists(completions_dir): mkdir(completions_dir)
    
    df = None 
    with jsonlines.open(t_file) as f:
        df = pd.DataFrame(f)

    df = df.astype(str)
    df = df.rename(columns={"prompt": "prompt", "completion": "expected_completion"})
    
    if randomize: df = df.sample(frac=1, random_state=1)  # shuffle df

    results_df = df.head(n) if n else df  # if n == None, use the full dev/test set
    results_df["expected_completion"] = results_df["expected_completion"].str.slice(0,-7)  # remove " <STOP>" from end
    results_df["actual_completion"] = "N/A" 

    return completions_dir, results_df


def model_completions(city: str, task: str, model: list, n: int = 10, body_prompt: bool = True, randomize: bool = True, final_model: bool = False):
    completions_dir, results_df = completion_prep(city, task, n, body_prompt, randomize, final_model)

    make_completions(completions_dir, model, n, results_df, randomize)


def multimodel_completions(city: str, task: str, models: list, n: int = 10, body_prompt: bool = True, randomize: bool = True, final_model: bool = False):
    # this is pretty inefficient on large dev/test sets due to the linear for loop, etc.
    # actual parallelization/threading would be complicated -- easy solution is to run completions for multiple 
    # models in different instances of terminal (not especially sophisticated, but works just fine)
    completions_dir, results_df = completion_prep(city, task, n, body_prompt, randomize, final_model)

    for m in models:
        make_completions(completions_dir, m, n, results_df, randomize)


if __name__ == "__main__":
    # ada_sizes = {5, 50, 500}
    # json_setup("chicago", only_body=False)
    # write_train_subfiles("chicago", "rent", ada_sizes)

    # sada_sizes = {5, 50, 500, 5000}
    # json_setup("seattle")
    # write_train_subfiles("seattle", "rent", sada_sizes)
    
    # upload_train_files("chicago")
    # upload_train_files("seattle") 

    datasize_models = ["ada:ft-lingmechlab:chicago-rent-5-2023-06-22-21-58-32",
                       "ada:ft-lingmechlab:chicago-rent-50-2023-06-22-22-39-13",
                       "ada:ft-lingmechlab:chicago-rent-500-2023-06-22-23-18-31",
                       "ada:ft-lingmechlab:chicago-rent-2023-06-23-00-20-18"]
    # multimodel_completions("chicago", "rent", datasize_models, n=None)

    tier_models = ["babbage:ft-lingmechlab:chicago-rent-2023-06-22-22-18-40",
                   "curie:ft-lingmechlab:chicago-rent-2023-06-22-22-41-58",
                   "davinci:ft-lingmechlab:chicago-rent-2023-06-23-00-08-29"]

    # multimodel_completions("chicago", "rent", tier_models, n=None)

    # model_completions("chicago", "rent", "davinci:ft-lingmechlab:chicago-rent-title-2023-06-23-05-48-03", n=None, body_prompt=False)

    stier_models = ["ada:ft-lingmechlab:seattle-rent-2023-06-23-22-08-58", 
                    "babbage:ft-lingmechlab:seattle-rent-2023-06-23-13-11-30", 
                    "curie:ft-lingmechlab:seattle-rent-2023-06-23-13-42-08"]
    
    sdatasize_models = ["ada:ft-lingmechlab:seattle-rent-5-2023-06-24-01-59-09", 
                        "ada:ft-lingmechlab:seattle-rent-50-2023-06-24-02-20-25",
                        "ada:ft-lingmechlab:seattle-rent-500-2023-06-24-02-29-21",
                        "ada:ft-lingmechlab:seattle-rent-5000-2023-06-24-04-01-38"]
    # multimodel_completions("seattle", "rent", stier_models, n=None)

                     
    sfinal_models = ["ada:ft-lingmechlab:seattle-race-2023-06-23-23-57-59",
                     "ada:ft-lingmechlab:seattle-rent-2023-06-23-22-08-58",
                     "ada:ft-lingmechlab:seattle-income-2023-06-24-01-56-14"]

    cprint("Nothing to do right now!", c="m")