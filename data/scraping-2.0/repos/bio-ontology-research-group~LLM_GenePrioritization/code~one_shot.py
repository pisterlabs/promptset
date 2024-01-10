#This code interacts with openAI's API to use GPT with one-shot prompting.
#run this code as: python one_shot.py prompt_index start_patient end_patient
#example:  python one_shot.py 0 1 101
import openai
import os
import pandas as pd
import time
import re
import sys

openai.api_key = 'your_OpenAI_API_key'

def get_completion(prompt, model="gpt-4"): 
        messages=[
    {
      "role": "system",
      "content": "// Role\nYou are an automated ranking system. You take a set of patient signs and symptoms (phenotypes) as input, as well as a set of genes in which a likely pathogenic variant has been identified using a bioinformatics system. You return a ranked list of genes according to the likelihood of the damaging variant in the gene causing the phenotypes of the patient.\n\n To do the ranking, first identify if there is any knowledge about mutations in the gene causing the same or similar phenotypes as observed in the patient. Use information about disease and phenotypes, animal models, gene functions, and anatomical site of expression. Automatically rank all genes on the last rank if no evidence exists, and rank all other genes based on the likelihood of causing the phenotypes.\n\n// Example\nuser\n A female patient who is suspected of having a genetic disease, presented with these clinical symptoms: \"Recurrent urticaria\", \"Recurrent abdominal pain\", \"Fatigue\", \"Fever\", \"arthralgia\", \"Lymphadenopathy\", \"Elevated circulating C-reactive protein\", \"Glomerulonephritis\", \"Elevated erythrocyte sedimentation rate\", \"Anemia\". Rank these genes according to their association with the symptoms of the patient:\"NKAPP1\", \"EXD2\", \"ENRICH2\", \"PDS5B\", \"CAMK2G\", \"DNASE1L3\", \"PCDH19\", \"ACADVL\", \"TRAF6P1\", \"CYP2T3P\" \nassistant\n{\"ranked_list\":\"1. DNASE1L3\n2. TRAF6P1\n3. ACADVL\n4. CAMK2G\n5. PCDH19\n6. ERICH2\n7. PDS5B\n8. EXD2\n9. NKAPP1\n10. CYP2T3P\n}"
    },
    {
      "role": "user",
      "content": prompt}]

        try:
                response = openai.ChatCompletion.create(model=model,messages=messages,temperature=0,)
                return response.choices[0].message["content"]
        except openai.error.OpenAIError as e:
                print ("OpenAI API error:", e)
                return get_completion(prompt, model)


#-------main-----------------------
for i in range(int(sys.argv[2]),int(sys.argv[3])):
        print ("Patient:"+str(i))
        f=open ("./questions/q_"+str(i)+".txt","r")
        prompt=f.readlines()[int(sys.argv[1])]
        f.close()
        response = get_completion(prompt)
        file = open("./responses_one_shot/gene"+str(int(sys.argv[1])*5+5)+"/r_"+str(i)+".txt", "w")
        if response is not None:
                gene_names = re.findall(r'\d+\.\s+([^:\s]+)[:\s]', response)
                numbered_list = "\n".join(f"{gene}\t{t+1}" for t, gene in enumerate(gene_names))
                file.write(numbered_list)

        else:
                file.write("Error")
        file.close()
