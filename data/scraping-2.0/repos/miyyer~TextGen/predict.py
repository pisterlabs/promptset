import os 
import math
import openai
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm 
from retry import retry

openai.api_key = os.getenv("OPENAI_API_KEY")

@retry(Exception, tries=3, delay=15)
def run_predict_module(data, model, max_tokens= 150, subset_ins=None, stop_token=' <EOS>', log_probs= 1):
    """
    Get predictions from a finetuned GPT3 model on a given dataset
    data: dataframe with its path 
    model: OpenAI Finetuned Model architecture 
    max_tokens: maximum number of tokens allowed in generation
    subset_nums: count of dataset instances to subset upon 
    """
    data = pd.read_csv(data)
    if subset_ins:
        data = data.iloc[:, ]
    ls_prompt     = []
    ls_prediction = []
    ls_gt         = []
    ls_ppl_prompt = []
    ls_ppl_resp   = []
    ls_ppl_total  = []


    for i in tqdm(range(len(data))):
        res = openai.Completion.create(model = model, prompt= data['prompt'][i], max_tokens= max_tokens, stop=[stop_token], logprobs= log_probs)

        completion = res['choices'][0]['text']
        completion = completion[1:]       # remove initial space
        prompt = data['prompt'][i][:-7]   # remove "\n\n###\n\n"

        ppl_response, ppl_total =  evaluate_response_perplexity(res, max_tokens)
        
        ls_prompt.append(data['prompt'][i])
        ls_prediction.append(completion)
        ls_gt.append(data['completion'][i])
        #ls_ppl_prompt.append(ppl_prompt)
        ls_ppl_resp.append(ppl_response)
        ls_ppl_total.append(ppl_total)
    
    predicted_df = pd.DataFrame({'Issue':ls_prompt, 'Actual_Solution':ls_gt, 'Pred_Solution':ls_prediction, 'Response_PPL':ls_ppl_resp, 'Total_PPL':ppl_total})

    return predicted_df

def calculate_perplexity(log_probs):
    """
    Calculates perplexity from aggregated log probabilities 
    """
    N = len(log_probs)
    return math.exp((-1/N) * np.sum(log_probs))

def evaluate_response_perplexity(response, max_tokens):
    """
    Evaluate perplexity of a given generated phrase from response
    """
    response_dict = dict(response['choices'][0])
    text = response_dict['text']

    log_probs = response_dict['logprobs']['token_logprobs'][1:]
    #log_probs_prompt = log_probs[:-max_tokens]
    log_probs_response = log_probs[-max_tokens:]
    
    #ppl_prompt = calculate_perplexity(log_probs_prompt)
    ppl_response = calculate_perplexity(log_probs_response)
    ppl_total = calculate_perplexity(log_probs)
    
    return  ppl_response, ppl_total



def main():
    """
    Store all the parser arguments and execute the prediction function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type= str, required=True, help= "Name of the finetuned GPT-3 model")
    parser.add_argument("--data", default=None, type= str, required= True, help= "Data File to carry out prediction on with path" )
    parser.add_argument("--max_tokens", default=150, type= int, required= False, help= "Maximum number of tokens allowed in prediction")
    parser.add_argument("--stop_token", default= " <EOS>", type=str, required=True, help= "Stop token format used in dataset")
    parser.add_argument("--subset_ins", default= None, required= False, type= int, help= "Subset the dataframe instances")
    parser.add_argument("--output_file", default= None, type= str, required= True, help= "Output File name with its location")
    parser.add_argument("--log_probs", default= 1, type = int, required= False, help= "returns n user specified most likely tokens")
    args = parser.parse_args()

    predicted_df = run_predict_module(args.data, args.model, args.max_tokens, args.subset_ins, args.stop_token, args.log_probs)
    predicted_df.to_csv(args.output_file, index= False)
    

if __name__ == '__main__':
    # Run the main function from here 
    main()








