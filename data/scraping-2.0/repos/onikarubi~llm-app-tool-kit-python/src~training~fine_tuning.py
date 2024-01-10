import os
import openai


def get_train_csv_file(csv_file):
    save_to = f"train_csv/{csv_file}"
    current_dir = os.path.abspath(os.path.dirname(__file__))
    save_to_file_path = os.path.join(current_dir, save_to)

    return save_to_file_path


# def create_dataset(csv_file):
#     try:
#         save_to_csv_path = get_train_csv_file(csv_file)
#         if os.path.exists(save_to_csv_path):
#             return

#         create_training_data(f"train_csv/{csv_file}")

#     except:
#         raise Exception("トレーニングデータ用のファイルを作成している最中にエラー発生")


# def upload_training_file():
#     return openai.File.create(
#         file=open("src/fine_tuning/train_json/yukkuri-marisa.jsonl"),
#         purpose="fine-tune",
#     )


# def fine_tuning_execute(upload_file_id):
#     response = openai.FineTuningJob.create(
#         training_file=upload_file_id,
#         model="gpt-3.5-turbo",
#         hyperparameters={"n_epochs": 4},
#     )
#     return response
