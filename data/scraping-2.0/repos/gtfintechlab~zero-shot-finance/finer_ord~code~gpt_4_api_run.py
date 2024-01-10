import openai,os,sys
import pandas as pd
from time import sleep, time
from datetime import date

sys.path.insert(0, '/home/research/git repos/zero-shot-finance')
from api_keys import APIKeyConstants


today = date.today()
openai.api_key = APIKeyConstants.OPENAI_API_KEY


start_t = time()
# load training data
test_data_path = "../data/test/test.csv"
data_df = pd.read_csv(test_data_path)

grouped_df = data_df.groupby(['doc_idx', 'sent_idx']).agg({'gold_label':lambda x: list(x), 'gold_token':lambda x: list(x)}).reset_index()
grouped_df.columns = ['doc_idx', 'sent_idx', 'gold_label', 'gold_token']



output_list = []
for index in range(grouped_df.shape[0]):
    token_list = grouped_df.loc[[index],['gold_token']].values[0, 0]
    label_list = grouped_df.loc[[index],['gold_label']].values[0, 0]
    sen = '\n'.join(token_list)

    message = "Discard all the previous instructions. Behave like you are an expert named entity identifier. Below a sentence is tokenized and each line contains a word token from the sentence. Identify 'Person', 'Location', and 'Organisation' from them and label them. If the entity is multi token use post-fix _B for the first label and _I for the remaining token labels for that particular entity. The start of the separate entity should always use _B post-fix for the label. If the token doesn't fit in any of those three categories or is not a named entity label it 'Other'. Do not combine words yourself. Use a colon to separate token and label. So the format should be token:label. \n\n" + sen

    prompt_json = [
            {"role": "user", "content": message},
    ]
    try:
        chat_completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=prompt_json,
                temperature=0.0,
                max_tokens=1000
        )
    except Exception as e:
        print(e)
        index = index - 1
        sleep(10.0)

    answer = chat_completion.choices[0].message.content

    output_list.append([label_list, sen, answer])
    
    sleep(6.0) 


results = pd.DataFrame(output_list, columns=["true_label", "original_sent", "text_output"])

time_taken = int((time() - start_t)/60.0)
results.to_pickle(f'../data/llm_prompt_outputs/gpt4_{today.strftime("%d_%m_%Y")}_{time_taken}')
