import openai
import time
import pandas as pd
import logging

openai.util.logging.getLogger().setLevel(logging.WARNING)

instructions = """We want to identify relationship of two given entities in a sentecne.

       For example, in the sentence 'Severe COVID-19 is associated with venous thromboembolic events and and immuno-thrombotic phenomena, responsible for pulmonary vascular damage.',
       given two entities are 'Severe COVID-19' and 'venous thromboembolic events', their relationship that should be returned is 'associated'.

       Another example is, in the sentence 'The shortening of hospital stays implies rethinking the pre- and post-operative management of lower limb arthroplasty.', 
       given two entities are 'hospital stays' and 'limb arthroplasty', their relationship that should be returned is 'implies'.


     """

target_base = """
       Now please extract entities on the following sentence:
       Result must be less than 3 words and must be a verb: "Relationship: ...."

     """


def get_result_df(df, key):
    openai.api_key = key
    result_df = pd.DataFrame(columns=['sentence', 'entity_1', 'entity_2', 'relationship'])

    for index in range(len(df)):
        sentence = df.iloc[index]['sentence']
        entity_1 = df.iloc[index]['entity_1']
        entity_2 = df.iloc[index]['entity_2']
        # original_relation = df.iloc[shift+index]['relationship']

        targets = target_base + sentence
        targets = targets + 'and two given entities are \'' + entity_1 + '\' and \'' + entity_2 + '\'.'

        prompt = instructions + targets
        response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0, max_tokens=2000)
        GPT3_result = response['choices'][0]['text']
        GPT3_result = GPT3_result[15:]

        result_df = result_df.append(
            {'sentence': sentence, 'entity_1': entity_1, 'entity_2': entity_2, 'relationship': GPT3_result},
            ignore_index=True)
        time.sleep(0.5)

    return result_df
