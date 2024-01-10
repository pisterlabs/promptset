import openai
from openai import cli
import time
import json
from sklearn.model_selection import train_test_split
# 這裡我們用 sklearn 的 train_test_split 來分割資料


# 記得要把 api_key 換成你自己的
openai.api_key = "2125ae816af7451fb5575493bf26251d"

# 你的資源名稱，格式為 https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_base = "https://kokochatgpt.openai.azure.com/"

# 你的資源類型，這裡我們用的是 Azure 的版本
openai.api_type = 'azure'

# 你的資源版本，這裡我們用的是 2022-12-01
openai.api_version = '2022-12-01'

# 輸出的訓練集和驗證集的檔案名稱
training_file_name = 'training.jsonl'
validation_file_name = 'validation.jsonl'

# 接著，把 data.json 裡的資料讀進來
sample_data = []
with open('data.json', 'r', encoding='utf-8') as file:  
    for line in file:  
        try:
            sample_data.append(json.loads(line.strip()))
        except:
            print(line)
            continue

# 把資料分成訓練集和驗證集
train_data, valid_data = train_test_split(
    sample_data, random_state=5566, train_size=0.8)

with open(training_file_name, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open(validation_file_name, "w", encoding="utf-8") as f:
    json.dump(valid_data, f, ensure_ascii=False, indent=4)

# 產生訓練集檔案
print(f'Generating the training file: {training_file_name}')
with open(training_file_name, 'w') as training_file:
    for entry in sample_data:
        json.dump(entry, training_file)
        training_file.write('\n')

# 產生驗證集檔案
print(f'Generating the validation file: {validation_file_name}')
with open(validation_file_name, 'w') as valid_file:
    for entry in sample_data:
        json.dump(entry, valid_file)
        valid_file.write('\n')

# 定義一個函式來檢查上傳的狀態
def check_status(training_id, validation_id):
    train_status = openai.File.retrieve(training_id)["status"]
    valid_status = openai.File.retrieve(validation_id)["status"]
    print(
        f'Status (training_file | validation_file): {train_status} | {valid_status}')
    return (train_status, valid_status)


# 把檔案上傳至 Azure OpenAI。
training_id = cli.FineTune._get_or_upload(training_file_name, True)
validation_id = cli.FineTune._get_or_upload(validation_file_name, True)

# 檢查上傳的狀態
(train_status, valid_status) = check_status(training_id, validation_id)

# 等待上傳完成
while train_status not in ["succeeded", "failed"] or valid_status not in ["succeeded", "failed"]:
    time.sleep(1)
    (train_status, valid_status) = check_status(training_id, validation_id)
