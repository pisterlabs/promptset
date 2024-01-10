import argparse
import pandas as pd
import os
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)

import openai
OPENAI_API_KEY="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"

openai.api_key=OPENAI_API_KEY

# input_dir="s3://nlgtest"
# formal_list = pd.read_pickle(os.path.join(input_dir,'formal_list_v5.pickle'))
# formal_list = formal_list['cos_ocr'] + formal_list['cos_sct'] + formal_list['cos_matt'] + formal_list['cos_ntd']
# df=pd.DataFrame()
# df["original keyword"]=formal_list
# df.to_csv(os.path.join(input_dir,"formal_list.csv"))



def get_gpt3_complete(keyword,max_tokens=15,temperature=0):
    response = openai.Completion.create(
      model="text-davinci-002",
      prompt=[f"given the key words below, generate a medical related only sentence ### \
      key words: intractable back pain -> sentence: the person has intractable back pain ***  \
      key words: at high risk -> sentence:  the person's condition has no change  *** \
      key words: 10 pain -> sentence:  the person has a rating of 10 pain  *** \
      key words: no change -> sentence:  the person's condition has no change *** \
      key words: pain is well controlled -> sentence:  the person control his pain ver well *** \
      key words: a rating of -> sentence:  the person has a rating of 10 pain level  *** \
      key words: good progress -> sentence:  the person has shown good progress in his condition *** \
      key words: {keyword} -> sentence: \
      "],
      temperature=0,
      max_tokens=max_tokens,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      stop=["\n","<|endoftext|>"]
    )
    return response


def main(formal_list, args):
    generated_sentence=[]
    original_keyword=[]
    
    for index,row in tqdm(formal_list.iterrows(), total=formal_list.shape[0]):
        response=get_gpt3_complete(keyword=row["original_keyword"],max_tokens=30,temperature=0)
        generated_sentence.append(response["choices"][0]['text'].strip("\n"))
        original_keyword.append(row["original_keyword"])
    output=pd.DataFrame()
    output["original keyword"]=original_keyword
    output["generated sentence"]=generated_sentence
    output.to_csv(os.path.join(args.output_dir,args.output_name))
    
if __name__=="__main__":
    argparser = argparse.ArgumentParser("generate sentence from keywords")
    
    argparser.add_argument('--chunk_num', type=int, default=20) 
    argparser.add_argument('--output_dir', type=str, default="s3://nlgtest") 
    argparser.add_argument('--idx', type=int, default=0)
    argparser.add_argument('--output_name', type=str, default=f"auto-complete-sentence.csv") 
    args = argparser.parse_args()
    print(args)
    
    args.output_name=f"auto-complete-sentence-v{args.idx}.csv"
    
    input_dir="s3://nlgtest"
    df=pd.read_csv(os.path.join(input_dir,"formal_list.csv"))
    df.drop(['Unnamed: 0'],axis=1,inplace=True)
    
    num=df.shape[0]//args.chunk_num
    
    if args.idx == args.chunk_num-1:
        data=df.iloc[args.idx*num:]
    else:
        data=df.iloc[args.idx*num:(args.idx+1)*num]
        
    
    main(data,args)
