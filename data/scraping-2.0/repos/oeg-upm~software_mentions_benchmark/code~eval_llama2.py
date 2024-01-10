import openai
import json
from os import listdir
from os.path import isfile, join
import pandas
import unicodedata


openai.api_base = "http://localhost:4891/v1"

openai.api_key = "not needed for a local LLM"

size_text = 2048


model = "llama-2-7b-chat"
#model = "gpt4all-falcon"

instructions = [
    {"role":"user","content":"As an unbiased labeler, please extract software mentions. Need follow these rules: Entities should be in the given text. Do not add or modify any words.  Separate multiple entities by ’|’ Only generate the output without any other words."},
    {"role":"user", "content":"Output if entities detected: {“name”: [name 1, name 2, ...]}. Output if not entities detected: {“name”: []}"},
]

def extract_software(message):

    results=[]

    #text = {"role":"user","content":"Text:"+message}

    #prompt0 = [
    #    {"role":"user","content":"As an unbiased labeler, please extract software mentions. Need follow these rules: Entities should be in the given text. Do not add or modify any words.  Separate multiple entities by ’|’ Only generate the output without any other words."},
    #    {"role":"user", "content":"Output if entities detected: {“name”: [name 1, name 2, ...]}. Output if not entities detected: {“name”: []}"},
    #    {"role":"user","content":"Text: "+message.replace("\n","")}
    #]

    #prompt1 = [
    #    {"role":"user","content":"As an unbiased labeler, please extract software mentions. Need follow these rules: Entities should be in the given text. Do not add or modify any words.  Separate multiple entities by ’|’ Only generate the output without any other words."},
    #    {"role":"user", "content":"Output if entities detected: {“name”: [name 1, name 2, ...]}. Output if not entities detected: {“name”: []}"},
    #    {"role":"user", "content":"In the text 'I am using Microsoft Excel for generating my datasets', Microsoft Excel is a software mention"},
    #    {"role":"user","content":"Text: "+message.replace("\n","")}
    #]
    
    prompt = [
        {"role":"user","content":"I would like you to behave as an unbiased labeler and extract software mentions in a given text. Follow the rules below: 1) Do not add or modify any words in the given text. 2) If you find multiple entities, separate them by '|',  3) Extract only software mentions, not developers, 4) The output should follow the format {\"name\":[software mention 1, software mention 2]}"},
        {"role":"user", "content":"Examples: Input: \"I am using Microsoft Excel for generating my datasets\" Output: {\"name\":[\"Microsoft excel\"]}"},
        {"role":"user", "content":"Input: \"SPSS is a package for doing statistical analysis over data\" Output: {\"name\":[\"SPSS\"]}"},
        {"role":"user", "content":"Input: \"Obama is a president of the United States\" Output: {\"name\": []}"},
        {"role":"user", "content":"Input: \"NumPy, SciPy, and Matplotlib are the foundations of this package, which is mostly written in Python.\" Output: {\"name\": [\"Numpy\", \"Scipy\", \"Matplotlib\"]}"},
        {"role":"user","content":"Text: "+message.replace("\n","")}
    ]
    '''
    prompt = [
        {"role":"user","content":"I would like you to behave as an unbiased labeler and extract software mentions in a given text. Follow the rules below: 1) Do not add or modify any words in the given text. 2) If you find multiple entities, separate them by '|',  3) Extract only software mentions, not developers, 4) The output should follow the format {\"name\":[software mention 1, software mention 2]}"},
        {"role":"user","content":"Text: "+message.replace("\n","")}
    ]
    '''

    response = openai.ChatCompletion.create(
        model = model,
        messages = prompt,
        max_tokens=50,
        temperature=0,
        top_p=0.95,
        n=1,
        echo=True,
        stream=False,
        reload=True
    )
    print("Response:")

    response_text = response["choices"][0]["message"]["content"]
    if response_text.find("Text:") > -1:

        response_text = response_text[response_text.index("Text:")::]
        if response_text.find("{") > -1 and response_text.find("}") > -1:
            response_text_filter = response_text[response_text.index("{"):response_text.index("}")+1].replace("\\","")
            if response_text_filter.find("[") > -1 and response_text_filter.find("]") > -1:
                response_text_filter = response_text_filter[response_text_filter.index("["):response_text_filter.index("]")+1].replace("\\","")
            else:
                print("Skip entity:"+str(response_text_filter))
                print(response_text)
                response_text_filter = []
        else:
            if response_text.find("[") > -1 and response_text.find("]") > -1:
                response_text_filter = response_text[response_text.index("["):response_text.index("]")+1].replace("\\","")
            else:
                response_text_filter = []

        print(response_text_filter)
        try:
            response_json = json.loads(response_text_filter)
        except Exception:
            print("Skip entity:"+str(response_text_filter))
            response_json = []
    else:
        print("No response detected")
        response_json = []

    return response_json

def partial_match(corpus, predictions):
    tp = []
    fp = []
    fn = []

    string_founded = False

    for x in corpus:
        for substring in predictions:
            if substring.find(x) >= 0:
                tp.append(x)
                string_founded = True
        if not string_founded:
            fn.append(x)
        else:
            string_founded = False

    string_founded = False

    for x in predictions:
        for substring in corpus:
            if x.find(substring) >= 0:
                string_founded = True
        if not string_founded:
            fp.append(x)
        else:
            string_founded = False

    return tp,fp,fn

def extract_string(filename):
    #open text file in read mode
    text_file = open(directory+"text-files/"+filename+".txt", "r", encoding="utf8")
    
    #read whole file to a string
    text = text_file.read()
    
    #close file
    text_file.close()
    

    return text

true_positives = 0
false_negatives = 0
false_positives = 0

true_positives_global = 0
false_negatives_global = 0
false_positives_global = 0

false_positives_list = []
false_negatives_list = []

round = 0
skip = 0

#directory = "datasets/corpus_research_software/benchmark/test-set/"
#directory="corpus/softcite/test-set/"
directory="corpus/softcite/test-set/"
data=pandas.read_csv(directory+"llama2-ner/annotations.tsv",sep='\t')


df = data.groupby(data["filename"])["span"].agg(list)

skipped_texts = 0

for item in df.items():
    print("****************")
    print("Round:"+str(round))
    print(item[0])
    print("-------------")
    text = extract_string(item[0])
    print("-------------")
    if len(text)<size_text:
        results_raw = extract_software(text)
        print("Corpus:"+str(item[1]))
        print("Prediction:"+str(results_raw))

        results = []
        for result in results_raw:
            if isinstance(result, int):
                results.append(str(result))
            else:
                results.append(result)
        

        #TRUE POSITIVES
        #result_tp = [x for x in item[1] if x in results]
        
        #FALSE NEGATIVES
        #result_fn = [x for x in item[1] if x not in results]
        
        #FALSE POSITIVES
        #result_fp = [x for x in results if x not in item[1]]

        result_tp,result_fp,result_fn = partial_match(item[1], results)

        false_positives_list.append({"file":item[0],"list":result_fp,"corpus":item[1],"predictions":results})
        false_negatives_list.append({"file":item[0],"list":result_fn,"corpus":item[1],"predictions":results})

        true_positives = len(result_tp)
        false_negatives = len(result_fn)
        false_positives = len(result_fp)
    else:
        skipped_texts+=1
        print("Text too long")
        true_positives = 0
        false_positives = 0
        false_negatives = 0




    #print("-------------")

    #print("True positives:"+str(true_positives))
    #print("False positives:"+str(false_positives))
    #print("False negatives:"+str(false_negatives))

    true_positives_global = true_positives_global + true_positives
    false_positives_global = false_positives_global + false_positives
    false_negatives_global = false_negatives_global + false_negatives


    if (true_positives_global == 0 and false_positives_global==0):
        precision = 0
    else:
        precision = true_positives_global / (true_positives_global+false_positives_global)

    if (true_positives_global == 0 and false_negatives_global==0):
        recall = 0
    else:
        recall = true_positives_global / (true_positives_global+false_negatives_global)

    if (precision == 0 and recall == 0):
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)


    round += 1

print("True positives (global):"+str(true_positives_global))
print("False positives (global):"+str(false_positives_global))
print("False negatives (global):"+str(false_negatives_global))

print("Precision:"+str(precision))
print("Recall:"+str(recall))
print("F1-score:"+str(f1_score))

print("False Negatives List")
print(false_negatives_list)

print("False Positives List")
print(false_positives_list)

results={"precision":precision,"recall":recall,"f1-score":f1_score,"processed_texts":round,"skipped_texts":skipped_texts,"false_positives":false_positives_list,"false_negatives":false_negatives_list}

results_json = json.dumps(results)

with open('eval_falcon_economics_2048_fewshoot_t0_prompt2_partialmatches', 'w',encoding="utf-8") as file:
    file.write(results_json)