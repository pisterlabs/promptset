import openai,os,sys
import pandas as pd
from time import sleep, time
from datetime import date
today = date.today()

openai.api_key = "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"


for seed in [5768, 78516, 944601]:  
    for data_category in ["lab-manual-split-combine"]:
        start_t = time()
        # load training data
        test_data_path = "../data/test/" + data_category + "-test" + "-" + str(seed) + ".xlsx"
        data_df = pd.read_excel(test_data_path)


        sentences = data_df['sentence'].to_list()
        labels = data_df['label'].to_numpy()

        output_list = []
        for i in range(len(sentences)): 
            sen = sentences[i]
            message = "Discard all the previous instructions. Behave like you are an expert sentence classifier. Classify the following sentence from FOMC into 'HAWKISH', 'DOVISH', or 'NEUTRAL' class. Label 'HAWKISH' if it is corresponding to tightening of the monetary policy, 'DOVISH' if it is corresponding to easing of the monetary policy, or 'NEUTRAL' if the stance is neutral. Provide the label in the first line and provide a short explanation in the second line. The sentence: " + sen

            prompt_json = [
                    {"role": "user", "content": message},
            ]
            try:
                chat_completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=prompt_json,
                        temperature=0.0,
                        max_tokens=1000
                )
            except Exception as e:
                print(e)
                i = i - 1
                sleep(10.0)

            answer = chat_completion.choices[0].message.content
            
            output_list.append([labels[i], sen, answer])
            sleep(1.0) 

        results = pd.DataFrame(output_list, columns=["true_label", "original_sent", "text_output"])

        time_taken = int((time() - start_t)/60.0)
        results.to_csv(f'../data/llm_prompt_outputs/gpt4_{data_category}_{seed}_{today.strftime("%d_%m_%Y")}_{time_taken}.csv', index=False)
