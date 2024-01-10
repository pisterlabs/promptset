# this was the script used for the token only approach on gpt
import openai
import time

# replace with actual api key
openai.api_key = "API-KEY"

def perform_api_request_with_retry(token, model, max_retries=8):
    retry = 0
    while retry < max_retries:
        try:
            completion = openai.ChatCompletion.create(
                model = model,
                messages=[
                    {"role":"system","content": "You are a Latin linguist and part-of-speech tagging expert. You are using UPOS (universal part of speech tags). UPOS tags are ADJ,ADP,ADV,AUX,CCONJ,DET,INTJ,NOUN,NUM,PART,PRON,PROPN,PUNCT,SCONJ,SYM,VERB and X. X stands for 'other'."},
                    {"role":"user",
                    "content":f"Do not tokenize the words further, they are already tokenized. Only output the tag (no explanations, no translations, no additional text). Return the UPOS tag for the token: {token}."}
                    ]
                )
            print(completion)
            return completion.choices[0].message.content
        except Exception as e:
            # print(f"Error: {e}")
            retry += 1
            # Implement exponential backoff with a sleep duration that increases with each retry
            sleep_duration = 2 ** retry
            # print(f"Retry {retry}, sleeping for {sleep_duration} seconds...")
            time.sleep(sleep_duration)
    return None  # If all retries fail


infiles =["../random-training-data-other-taggers/test_tok_ittb.txt","../random-training-data-other-taggers/test_tok_llct.txt",
        "../random-training-data-other-taggers/test_tok_udante.txt","../random-training-data-other-taggers/test_tok_proiel.txt",
        "../random-training-data-other-taggers/test_tok_perseus.txt"]

outfiles =["../random-training-data-other-taggers/gpt4_ittb.txt","../random-training-data-other-taggers/gpt4_llct.txt",
        "../random-training-data-other-taggers/gpt4_udante.txt","../random-training-data-other-taggers/gpt4_proiel.txt",
        "../random-training-data-other-taggers/gpt4_perseus.txt"]
# Open your output file
for i,file in enumerate(infiles):
    with open(file, "r") as infile:
        token_list = infile.readlines()
    with open(outfiles[i], "w") as out:
        for token in token_list:
            if token == "" or token == "\n" or token == " ":
                continue
            else:
                result = perform_api_request_with_retry(token, model="gpt-4")
                if result:
                    if len(result.split())>1:
                        result = perform_api_request_with_retry(token, model="gpt-4")
                        
                    row = token.strip() +"\t" +result.strip()+"\n"
                    out.write(row)






