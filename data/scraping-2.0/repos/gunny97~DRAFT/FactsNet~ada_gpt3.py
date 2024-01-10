import pandas as pd
import os
import openai
from evaluate import load


f1_metric = load('f1')
acc_metric = load('accuracy')
rec_metric = load('recall')
prec_metric = load('precision')

openai.api_key = "xxxxxxxxxx"

restart_sequence = "\n"

def gpt3_clf(prompt, content_name):

    if 'facts' in content_name:
        content_name = content_name.replace('-facts','')
        
    if '-' in content_name:
        content_name = content_name.replace('-',' ')
    
    response = openai.Completion.create(
    # model="text-ada-001",
    model = 'ada',
    prompt=f"Is the following text related to {content_name}? \
            Answer yes or no. \
            \n\n\n\"{prompt}\"",
    temperature=0,
    max_tokens=6,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def load_testdata_path():
    test_data_path = "/crawl/crawler/test_data"
    test_data_path_list = []

    for cat in os.listdir(test_data_path):
        sub_path = os.path.join(test_data_path,cat)

        for cat_2 in os.listdir(sub_path):
            edge_path = os.path.join(sub_path, cat_2)
            test_data_path_list.append(edge_path)
    return test_data_path_list

def main(content_name, test_data_path, few_shot, query_list):
    test_set = pd.read_csv(test_data_path)
    test_set.drop(['Unnamed: 0'], axis=1, inplace=True)
    test_set.columns = ['text', 'label']

    # positive samples
    for i in range(test_set.shape[0]):
        if test_set.iloc[i]['label'] == 0:
            test_set.label[i] = 1
        elif test_set.iloc[i]['label'] == 1 or test_set.iloc[i]['label'] == 2:
            test_set.label[i] = 0

    test_set.drop(index = len(test_set)-1, inplace=True)

    if few_shot:
        gpt3_pred = []
        for i in range(len(test_set)):

            text = test_set.iloc[i]['text']
            
            query_prompt = []
            for q in query_list:
                query_prompt.append("'"+str(q)+ "'" + ": yes\n")

            query_prompt_str = "\n" + "".join(query_prompt)

            text = query_prompt_str + '\n' + '\n' + '\n' + "'"+str(text)+ "'" + ":"

            resp = gpt3_clf(text, content_name).lower()
            if 'yes' in resp:
                gpt3_pred.append(1)
            else:
                gpt3_pred.append(0)

        f1_score = f1_metric.compute(predictions=gpt3_pred, references=test_set['label'])
        acc_score = acc_metric.compute(predictions=gpt3_pred, references=test_set['label'])
        rec_score = rec_metric.compute(predictions=gpt3_pred, references=test_set['label'])
        prec_score = prec_metric.compute(predictions=gpt3_pred, references=test_set['label'])

        return f1_score, acc_score, rec_score, prec_score

    else:
        gpt3_pred = []
        for i in range(len(test_set)):

            text = test_set.iloc[i]['text']
            resp = gpt3_clf(text, content_name).lower()
            if 'yes' in resp:
                gpt3_pred.append(1)
            else:
                gpt3_pred.append(0)

        f1_score = f1_metric.compute(predictions=gpt3_pred, references=test_set['label'])
        acc_score = acc_metric.compute(predictions=gpt3_pred, references=test_set['label'])
        rec_score = rec_metric.compute(predictions=gpt3_pred, references=test_set['label'])
        prec_score = prec_metric.compute(predictions=gpt3_pred, references=test_set['label'])

        return f1_score, acc_score, rec_score, prec_score


if __name__ == "__main__":

    import logging

    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()

    few_shot = True
    for shot_number in [1,3,5,0]:

        shot = 'few-shot'

        if shot_number == 0:
            few_shot = False
            shot = 'zero-shot'

        fileHandler = logging.FileHandler(f'./0103_ORIGINAL_GPT3_ADA_factsNet_{shot}_{shot_number}-shot_results.log')
        
        logger.addHandler(streamHandler)
        logger.addHandler(fileHandler)
        
        query_path = "/crawl/crawler/query_output"
        query_path_list = []


        for cat in os.listdir(query_path):
            if cat == 'general':
                continue
            else:
                sub_path = os.path.join(query_path,cat)

                for cat_2 in os.listdir(sub_path):
                    edge_path = os.path.join(sub_path, cat_2)
                    for cat_3 in os.listdir(edge_path):
                        file_path = os.path.join(edge_path, cat_3)
                        query_path_list.append(file_path)

        

        test_data_path_list = load_testdata_path()

        for iii, test_data_path in enumerate(test_data_path_list):
            
            content_name = test_data_path.split('/')[-1].replace('.csv','')


            if few_shot:
                for q in query_path_list:
                    # if content_name in q:
                    if content_name == q.split('/')[-1].replace('query_','').replace('.csv',''):
                        content_query_path = q

                query_list = list(pd.read_csv(content_query_path)['query'])
                print("="*30,content_name,"="*30)
                
                query_list = query_list[:shot_number]

                f1, acc, rec, prec = main(content_name, test_data_path,few_shot, query_list)

                logger.setLevel(level=logging.DEBUG)
                logger.debug(f"content_name: {content_name}-{shot_number}")
                logger.debug(f"test_data_path: {test_data_path}")
                logger.debug(f"content_query_path: {content_query_path}")
                logger.debug(f"f1-score: {round(f1['f1'],4)}")
                logger.debug(f"accuracy: {round(acc['accuracy'],4)}")
                logger.debug(f"recall: {round(rec['recall'],4)}")
                logger.debug(f"precision: {round(prec['precision'],4)}")
                logger.debug("="*100)

            else:
                print("="*30,content_name,"="*30)

                f1, acc, rec, prec = main(content_name, test_data_path,few_shot, None)

                logger.setLevel(level=logging.DEBUG)
                logger.debug(f"content_name: {content_name}")
                logger.debug(f"test_data_path: {test_data_path}")
                logger.debug(f"content_query_path: {content_query_path}")
                logger.debug(f"f1-score: {round(f1['f1'],4)}")
                logger.debug(f"accuracy: {round(acc['accuracy'],4)}")
                logger.debug(f"recall: {round(rec['recall'],4)}")
                logger.debug(f"precision: {round(prec['precision'],4)}")
                logger.debug("="*100)        
        
        fileHandler.close()
        logger.removeHandler(fileHandler)