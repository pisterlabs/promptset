import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="# Python 3.7\n \ndef randomly_split_dataset(folder, filename, split_ratio=[0.8, 0.2]):\n    df = pd.read_json(folder + filename, lines=True)\n    train_name, test_name = \"train.jsonl\", \"test.jsonl\"\n    df_train, df_test = train_test_split(df, test_size=split_ratio[1], random_state=42)\n    df_train.to_json(folder + train_name, orient='records', lines=True)\n    df_test.to_json(folder + test_name, orient='records', lines=True)\nrandomly_split_dataset('finetune_data/', 'dataset.jsonl')\n    \n# An elaborate, high quality docstring for the above function:\n\"\"\"",
  temperature=0,
  max_tokens=150,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["#", "\"\"\""]
)

# Prompt
# # Python 3.7
 
# def randomly_split_dataset(folder, filename, split_ratio=[0.8, 0.2]):
#     df = pd.read_json(folder + filename, lines=True)
#     train_name, test_name = "train.jsonl", "test.jsonl"
#     df_train, df_test = train_test_split(df, test_size=split_ratio[1], random_state=42)
#     df_train.to_json(folder + train_name, orient='records', lines=True)
#     df_test.to_json(folder + test_name, orient='records', lines=True)
# randomly_split_dataset('finetune_data/', 'dataset.jsonl')
    
# # An elaborate, high quality docstring for the above function:
# """
# Sample response
# """ This function randomly splits a dataset into two parts, a training set and a test set, according to a given split ratio.

# Parameters:
#     folder (str): The path to the folder containing the dataset.
#     filename (str): The name of the dataset file.
#     split_ratio (list): A list of two floats representing the ratio of the training set and the test set.

# Returns:
#     None
# """