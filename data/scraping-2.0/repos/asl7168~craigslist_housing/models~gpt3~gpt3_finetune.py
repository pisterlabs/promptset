import openai
from openai_credentials import skey, org
import json 
from os import path, mkdir

openai.api_key = skey
openai.organization = org if org else None

chicago_train_files = {}
with open("./training_logs/chicago_train_files.json") as f:
    chicago_train_files = json.load(f)

seattle_train_files = {}
with open("./training_logs/seattle_train_files.json") as f:
    seattle_train_files = json.load(f)

train_files = {"chicago": chicago_train_files, "seattle": seattle_train_files}


def update_finetune_log(c: str, t: str, m: str, n: int, b: bool, v):
    b_str = "body" if b else "title"

    data = {}
    if path.exists("./training_logs/finetunes.json"):
        with open("./training_logs/finetunes.json") as f:
            data = json.load(f)

    def update_data():
        # print(data)
        if c in data:
            if t in data[c]:
                if m in data[c][t]:
                    if b_str in data[c][t][m]:
                        if n: data[c][t][m][b_str][str(n)] = v
                        else: data[c][t][m][b_str]["FULL"] = v 
                    else:
                        data[c][t][m][b_str] = {}
                        update_data()
                else:
                    data[c][t][m] = {}
                    update_data()
            else:
                data[c][t] = {}
                update_data()
        else:
            data[c] = {}
            update_data()
    
    update_data()
    with open("./training_logs/finetunes.json", "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)


def train(city: str, task: str, model: str = "ada", n: int = None, body_prompt: bool = True):
    n_str = f"_{n}" if n else ""
    prompt_str = "" if body_prompt else "_TITLE"
    filename = f"{city}_{task}_train{prompt_str}{n_str}.jsonl"
    suffix = f"{city}_{task}_{prompt_str}{n_str}"
    train_id = train_files[city][filename]
    
    response = openai.FineTune.create(training_file=train_id, model=model, suffix=suffix)
    # print(response)
    update_finetune_log(city, task, model, n, body_prompt, response)
    
    return response["id"]


def rent_train(city: str, model: str = "ada", n: int = None, body_prompt: bool = True):
    return train(city, "rent", model, n, body_prompt)


def income_train(city: str, model: str = "ada", n: int = None, body_prompt: bool = True):
    return train(city, "income", model, n, body_prompt)


def race_train(city: str, model: str = "ada", n: int = None, body_prompt: bool = True): 
    return train(city, "race", model, n, body_prompt)
 
def get_finetune_events():
    finetune_log = None 
    with open("./training_logs/finetunes.json") as f:
        finetune_log = json.load(f)

    data = {}
    for c in finetune_log.keys():
        data[c] = {}
        for t in finetune_log[c].keys():
            data[c][t] = {}
            for m in finetune_log[c][t].keys():
                data[c][t][m] = {}
                for b in finetune_log[c][t][m].keys():
                    data[c][t][m][b] = {}
                    for n in finetune_log[c][t][m][b].keys():
                        data[c][t][m][b][n] = openai.FineTune.list_events(id=finetune_log[c][t][m][b][n]["id"])

    with open("./training_logs/events.json", "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)

""" 
TODO: add code for continuing training on a different file if necessary. Hopefully don't need this because 
it'd make everything slightly more complicated. First, would need to update train() to ensure it uses
city_rent_train_n_0.jsonl to start. Then would make a resume function to do the next file, like:
1. open the finetune_logs
2. ft_log = data[c][m][n/FULL]
3. last_training_file = ft_log["training_files"]["filename"]
4. new_filenum = last_training_file[-7] + 1 
5. new_training_file = f"{city}_task_train{n_suffix}_{new_filenum}.jsonl"
6. train_id = train_files[city][filename]
7. ft_id = ft_log["id"]
8. fts = openai.FineTune.list()["data"]
9. ft_modelname = [fts[i]["fine_tuned_model"] for i in range(len(fts)) if fts[i]["id"] == ft_id][0]
        DIFFERENT FROM model PROVIDED TO FUNCTION -- BUT WE STILL NEED model FOR ft_log KEYS
10. response_dict = openai.FineTune.create(training_file=train_id, model=ft_modelname) 
        NO SUFFIX SINCE CONTINUING -- https://platform.openai.com/docs/guides/fine-tuning/advanced-usage
11. update_finetune_log(city, model, n, response_dict)
12. return response_dict["id"]
Not sure this would *exactly* work but I think it gets everything that's needed?
"""

if __name__ == "__main__":
    # rent_train("chicago", n=5)
    # rent_train("chicago", n=50)
    # rent_train("chicago", n=500)
    # rent_train("chicago")

    # rent_train("chicago", "babbage")
    # rent_train("chicago", "curie")
    # rent_train("chicago", "davinci")
    
    # rent_train("chicago", "davinci", body_prompt=False)
    
    # rent_train("seattle", "ada")
    # rent_train("seattle", "babbage")
    # rent_train("seattle", "curie")

    # TODO: FINAL TASK TRAINS
    # rent_train("seattle")
    # rent_train("seattle", n=5)
    # rent_train("seattle", n=50)
    # rent_train("seattle", n=500)
    # rent_train("seattle", n=5000)

    # race_train("seattle")
    # income_train("seattle")
    
    get_finetune_events()