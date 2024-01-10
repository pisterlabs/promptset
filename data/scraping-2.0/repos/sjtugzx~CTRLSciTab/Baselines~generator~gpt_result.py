import os
import openai
import requests

import datetime
import time




# from evaluate import evaluate_meteor, evaluate_bert_score, evaluate_bleurt, ctrlsum_eval



openai.api_key = "YOUR_API_KEY"


def get_chatgpt_response(messages):
    resp = requests.post(url="https://api.openai.com/v1/chat/completions",
                         headers={
                             "Content-Type": "application/json",
                             "Authorization": f"Bearer {openai.api_key}"
                         },
                         json={
                             "model": "gpt-3.5-turbo",
                             "messages": messages,
                         }
    )
    # 解析响应
    if resp.status_code == 200:
        data = resp.json()
        text = data["choices"][0]["message"]
        return text
    else:
        print(resp.json())
        return "Sorry, something went wrong."

def get_gpt3_response(messages):
    API_URL = "https://api-inference.huggingface.co/models/Nicki/gpt3-base"
    API_TOKEN = "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"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    response = requests.post(API_URL, headers=headers, json={
        "inputs": messages
    })
    return response.json()

def get_LLaMa7b_response(messages):
    API_URL = "https://api-inference.huggingface.co/models/decapoda-research/llama-7b-hf"
    headers = {"Authorization": "Bearer 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"}

    response = requests.post(API_URL, headers=headers, json={
        "inputs": messages
    })
    return response.json()

def read_resource_file(file_path):
    print("read source file!")
    resource_list = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            new_dict={}
            new_dict["role"]="assistant"
            new_dict["content"]=line.strip()
            resource_list.append(new_dict)

    return resource_list

def write_output_file(file_path, predicted_list):
    with open(file_path, 'w') as f:
        f.writelines("%s\n" % item for item in predicted_list)



if __name__ == "__main__":
    resource_list = read_resource_file("../retriever/tsdae_without_bginfo/v2/test.source")
    response_list = []
    idx = 0

    with open("../retriever/tsdae_without_bginfo/v2/testwob.chatgpt", "a") as f:
        for message in resource_list:
            response = get_chatgpt_response([message])['content'].replace("\n", " ")
            # print(response)
            response_list.append(response)
            idx+=1
            f.write(response + "\n")
            print("finish writing the "+str(idx)+"th record into the file.")

    print("finish writing the " + str(idx) + "th record into the file.")

    ################################################################################################
    # basic evaluations

    # evaluation_text = []
    #
    # print('Basic Evaluations (BLEU, METEOR, BERTScore, BLEURT)')
    # evaluation_text += 'Basic Evaluations (BLEU, METEOR, BERTScore, BLEURT)' + '\n'
    # # bleu scores
    # bleu_info = ctrlsum_eval("../outputs/inference_results/bart_generation_output.txt", "../retriever/tsdae_data/v2/test.target")
    # print('BLEU score: %.4f' % bleu_info)
    # evaluation_text += 'BLEU score: %.4f' % bleu_info + '\n'
    # #
    # predicted_list = response_list
    # reference_list = []
    # with open("../retriever/tsdae_data/v2/test.target", 'r', encoding='utf8') as f:
    #     for line in f:
    #         reference_list.append(line.strip())

    # reference_list=reference_list[:2]

   # meteor scores
   #  meteor_info = evaluate_meteor(predicted_list, reference_list)
   #  print('METEOR score: %.4f' % meteor_info)
   #  evaluation_text += 'METEOR score: %.4f' % meteor_info + '\n'

    # # bert scores
    # bert_score_precision, bert_score_recall, bert_score_f1 = evaluate_bert_score(
    #     predicted_list, reference_list)
    #
    # print('bert_score precision is %5f, bert_score recall is %5f, bert_score f1 is %5f' % (
    #     bert_score_precision, bert_score_recall, bert_score_f1))
    # evaluation_text += 'bert_score precision is %5f, bert_score recall is %5f, bert_score f1 is %5f' % (
    #     bert_score_precision, bert_score_recall, bert_score_f1) + '\n'

    # bleurt scores
    # bleurt_info = evaluate_bleurt(checkpoint_path=
    #                               "../../../../bleurt/bleurt/test_checkpoint", predicted_list
    #                               =predicted_list, reference_list=reference_list)
    # print('BLEURT score: %.4f' % bleurt_info)
    # evaluation_text += 'BLEURT score: %.4f' % bleurt_info + '\n'



    # print('----------------------------------------------------------------')
    # with open("../retriever/tsdae_data/v2/test.chatgptscore", 'w') as f:
    #     f.writelines(evaluation_text)
    ###############################################################################################



    ################################################################################################
    # basic evaluations

    # evaluation_text = []
    #
    # print('Basic Evaluations (BLEU, METEOR, BERTScore, BLEURT)')
    # evaluation_text += 'Basic Evaluations (BLEU, METEOR, BERTScore, BLEURT)' + '\n'
    # # bleu scores
    # bleu_info = ctrlsum_eval("../outputs/inference_results/bart_generation_output.txt", "../retriever/tsdae_data/v2/test.target")
    # print('BLEU score: %.4f' % bleu_info)
    # evaluation_text += 'BLEU score: %.4f' % bleu_info + '\n'
    # #
    # predicted_list = response_list
    # reference_list = []
    # with open("../retriever/tsdae_data/v2/test.target", 'r', encoding='utf8') as f:
    #     for line in f:
    #         reference_list.append(line.strip())
    #
    # reference_list = reference_list[:2]
    #
    # # meteor scores
    # meteor_info = evaluate_meteor(predicted_list, reference_list)
    # print('METEOR score: %.4f' % meteor_info)
    # evaluation_text += 'METEOR score: %.4f' % meteor_info + '\n'
    #
    # # # bert scores
    # # bert_score_precision, bert_score_recall, bert_score_f1 = evaluate_bert_score(
    # #     predicted_list, reference_list)
    # #
    # # print('bert_score precision is %5f, bert_score recall is %5f, bert_score f1 is %5f' % (
    # #     bert_score_precision, bert_score_recall, bert_score_f1))
    # # evaluation_text += 'bert_score precision is %5f, bert_score recall is %5f, bert_score f1 is %5f' % (
    # #     bert_score_precision, bert_score_recall, bert_score_f1) + '\n'
    #
    # # bleurt scores
    # bleurt_info = evaluate_bleurt(checkpoint_path=
    #                               "../../../../bleurt/bleurt/test_checkpoint", predicted_list
    #                               =predicted_list, reference_list=reference_list)
    # print('BLEURT score: %.4f' % bleurt_info)
    # evaluation_text += 'BLEURT score: %.4f' % bleurt_info + '\n'
    #
    # print('----------------------------------------------------------------')
    # with open("../retriever/tsdae_data/v2/test.chatgptscore", 'w') as f:
    #     f.writelines(evaluation_text)
    # ###############################################################################################
    #
