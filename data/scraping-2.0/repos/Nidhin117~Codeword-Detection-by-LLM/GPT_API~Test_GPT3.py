import pandas as pd
import openai
import re


df =  pd.read_csv('./BigQuery/Files/CombinesDS_2.txt', delimiter='|')
dataset_size = 0.5
file = open("API_Key.txt",'r')
API_KEY=file.readline()
openai.api_key = API_KEY
model_name = "gpt-3.5-turbo"
log_fname = "GPT_test_log2.txt"
result_file = "gpt_test_result2.txt"

fewshot_msg=[
        {"role": "system", "content": "You are an investigator looking for code word for drugs hidden in sentences. Learn from the given examples and answer with a No if code word for drugs is not present. If code word for drugs is present, identify the code word and what it refers to"},
        {"role": "user", "content": "Lol, that shit is probably so stepped on you can't even call it coke anymore."},
        {"role": "assistant", "content": "Present: Yes, Code word : coke, Code word meaning : Cocaine"},
        {"role": "user", "content": "No one would resist a pot of soup"},
        {"role": "assistant", "content": "Present: No"}
    ]

def format_msg(msg):
    msg = msg.rstrip().replace("'","")
    if msg[-1] != ".":
        msg += "."
    return msg

def call_gpt(fewshot_msg):
    global model_name
    err_flag = False
    try:
        response = openai.ChatCompletion.create(
        model=model_name,
        messages=fewshot_msg,
        temperature=0,
        )
        return response["choices"][0]["message"]["content"],err_flag
    except openai.error.APIError as e :
        err_flag = True
        err_msg = f"API error {e}"
    except openai.error.APIConnectionError:
        err_flag = True
        err_msg = f"APIConnectionError error {e}"
    except openai.error.InvalidRequestError as e:
        err_flag = True
        err_msg = f"InvalidRequestError error {e}"
    except openai.error.Timeout as e:
        err_flag = True
        err_msg = f"Timeout error {e}"
    except openai.error.AuthenticationError as e:
        err_flag = True
        err_msg = f"AuthenticationError error {e}"
    except openai.error.PermissionError as e:
        err_flag = True
        err_msg = f"PermissionError error {e}"
    except openai.error.RateLimitError as e:
        err_flag = True
        err_msg = f"RateLimitError error {e}"
    if err_flag:
        return err_msg,err_flag
    else:
        return "Encountered unhandled exception",err_flag



def logger(fname,msg):
    file= open(fname,'a',encoding='utf-8')
    file.write(f'{msg} \n')
    file.close()

def init_write(fname, text):
    file= open(fname,'w',encoding='utf-8')
    file.write(text + '\n')
    file.close()

def log(msg):
    global log_fname
    logger(log_fname, msg)


init_write(log_fname,"Logging starts")
init_write(result_file," ")


counter = 0 
prompt_msg = ""
exception_counter = 0 
pattern_yes =  r'Present\s*:\s*yes*\s*\,\s*Code word\s*:\s*\w*\s*\w*\s*\,\s*Code word meaning\s*:\s*\w*\s*\w*\s*'
pattern_no = r'Present\s*:\s*no\s*'
for index,row in df.iterrows():
    
    if index == (int(len(df)*dataset_size)): # stop at 50% of dataset
    #if index == 5: 
        break
    log (f" Sentence: {row['Sentences']}")
    counter += 1
    prompt_msg =  f"{ format_msg(row['Sentences'])} ?" 
    task_prompt = {"role": "user", "content": prompt_msg }
    fewshot_msg.append(task_prompt) 
    #log(f"{fewshot_msg}")
 
    result,err_flg = call_gpt(fewshot_msg)
    log (result)
    
    if not err_flg:
        try:
            match_yes = re.fullmatch(pattern_yes,result,re.IGNORECASE)
            match_no = re.fullmatch(pattern_no,result,re.IGNORECASE)
            if match_yes or match_no:
                val = result.split(",")
                present = val[0].split(":")[1]
                #print(present)
                if present.lower().strip() == "yes":
                    code_word = val[1].split(":")[1]
                    cw_meaning = val[2].split(":")[1]
                    logger(result_file,f"{row['Sentences']} | {present} | {code_word} | {cw_meaning}")
                else:
                    logger(result_file,f"{row['Sentences']} | {present} | NA | NA")
            else:
                log("Response pattern is not as expected")
                logger(result_file,f"{row['Sentences']} | Pattern not matching | NA | NA")
                
        except Exception as e:
            log(f"In exception due to {e}")
            exception_counter += 1
            logger(result_file,f"{row['Sentences']} | In exception {e} | NA | NA")
    else:
        exception_counter += 1
        log(f"In API exception{result}")
        logger(result_file,f"{row['Sentences']} | In API exception {result} | NA | NA")
    if exception_counter == 10:
        log("Encountered too many exceptions, going to exit")
        break
     
    fewshot_msg.pop()

print("Done with execution")

    



