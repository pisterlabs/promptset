import streamlit as st
import pandas as pd
import openai
import csv
import pandas
import os
from tqdm import tqdm
import re
import time
import random


openai.api_key = st.secrets["openai"]["OPENAI_API_KEY"]

def chat(system, user_assistant):
    assert isinstance(system, str), "`system` should be a string"
    assert isinstance(user_assistant, list), "`user_assistant` should be a list"
    system_msg = [{"role": "system", "content": system}]
    user_assistant_msgs = [
      {"role": "assistant", "content": user_assistant[i]} if i % 2 else {"role": "user", "content": user_assistant[i]}
      for i in range(len(user_assistant))]

    msgs = system_msg + user_assistant_msgs
  #for delay_secs in (2**x for x in range(0, 6)):
    try:
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                          messages=msgs,
                                          temperature=0,  # Control the randomness of the generated response
                                          n=1,  # Generate a single response
                                          stop=None )
    except openai.OpenAIError as e:
        st.error("OpenAI server is overloaded. Please try after sometime.")
    #    randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
    #    sleep_dur = delay_secs + randomness_collision_avoidance
    #    print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
    #    time.sleep(sleep_dur)
    #    continue
    status_code = response["choices"][0]["finish_reason"]
    assert status_code == "stop", f"The status code was {status_code}."
    return response["choices"][0]["message"]["content"]

def pii_redact(input):

    pat1 = r'[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+.com' #email
    pat2 = r'[A-Z]{5}[0-9]{4}[A-Z]{1}'            #PAN number
    pat3 = r'[0-9]'                               #phone number, aadhar , amount , date, policy number
    combined_pat = r'|'.join((pat1, pat2, pat3))
    redact_out = re.sub(combined_pat,'',input)
    return redact_out

def detect_lang(df):
    progress_bar = st.progress(0)
    status_count = st.empty()
    processed_rows = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):
        input_line = pii_redact(row['INPUT'])
        response_fn_test = chat(''' identify the language of given sentence. strictly don't justify your answer.Don’t give information not mentioned in the CONTEXT INFORMATION.
                                    ."strictly provide output in one word" 
                                             For example if the given statement is in english then just give output as - "ENGLISH" , 
                               ''',['''sentence is  - ''' + '''"''' + input_line + '''"'''+ '''. 
         '''])
        
        df.at[index, 'DETECTED_LANGUAGE'] = response_fn_test.replace('.', '').strip().upper()
        df.at[index, 'REDACTED_INPUT'] = input_line
        processed_rows += 1
        progress_bar.progress((index + 1) / len(df))
        status_count.write(f"Detecting language {processed_rows}/{len(df)}")
    progress_bar.empty()
    status_count.empty()

    return df

def translate(df,cat_list):
        lang_detected_df = detect_lang(df)
        progress_bar = st.progress(0)
        status_count = st.empty()
        processed_rows = 0
        #for index, row in df.iterrows():
        for index, row in tqdm(lang_detected_df.iterrows(), total=len(lang_detected_df)):
            input_line = row['REDACTED_INPUT']
            op = row['DETECTED_LANGUAGE']
            lang = str(op).lower()
            if lang != 'english':
                response_fn_test = chat(''' You are a language translator.Translate the given statement to english
                                        Don’t justify your answers. Don’t give information not mentioned in the CONTEXT INFORMATION.
                                        Don't give subheadings. only provide translated output''',['''translate the statement - ''' + '''"''' + input_line + '''"'''+ '''. 
                                            '''])
            #st.write(f"translated output: {response_fn_test}")
            #row['TRANSLATION'] = response_fn_test
            else:
                response_fn_test = "Statement is not translated"
            lang_detected_df.at[index, 'TRANSLATION'] = response_fn_test.replace('"', '')
            processed_rows += 1
            progress_bar.progress((index + 1) / len(lang_detected_df))
            status_count.write(f"Translating row {processed_rows}/{len(lang_detected_df)}")
        progress_bar.empty()
        status_count.empty()
        #st.write(lang_detected_df)
        df_final = classify(lang_detected_df,cat_list)
        #st.write(df)
        return df_final

def classify(df,cat_list):
        progress_bar = st.progress(0)
        status_count = st.empty()
        processed_rows = 0
        #for index, row in df.iterrows():
        for index, row in tqdm(df.iterrows(), total=len(df)):
            if row['TRANSLATION'] == "Statement is not translated":
                input_line = row['REDACTED_INPUT']
            else:
                input_line = row['TRANSLATION']
            response_fn_test = chat(''' You are a language classifier.
        below are the only list of categories available to you and you must only use any one of the below categories to classify: [''' + ', '.join(cat_list) + '''] 
         only provide classified category name as output along with confidence score ranging 
        from 0 to 1.output should be like - [category_name,confidence_score]. If statement cannot be classified give output as "unknown".
        Don’t justify your answers.Don't give subheadings. Don’t give information not mentioned in the CONTEXT INFORMATION.
        
        ''',['''classify the statement - ''' + input_line ])
            #st.write(f"translated output: {response_fn_test}")
            #row['TRANSLATION'] = response_fn_test
            if response_fn_test.strip().lower() != 'unknown':
                resp_trans = response_fn_test.replace('"', '').replace('[', '').replace(']', '').split(',')
                df.at[index, 'CATEGORY'] = resp_trans[0]
                df.at[index, 'CONFIDENCE_SCORE'] = resp_trans[1]
            else:
                df.at[index, 'CATEGORY'] = response_fn_test
                df.at[index, 'CONFIDENCE_SCORE'] = 0
            processed_rows += 1
            progress_bar.progress((index + 1) / len(df))
            status_count.write(f"Classifying row {processed_rows}/{len(df)}")
        progress_bar.empty()
        status_count.empty()
        #st.write(df2)
        return df

def main():
    st.image('./images/logo-removebg-preview.png')
    #"st.session_state object:",st.session_state

    if st.session_state.get("login_token") != True:
        st.error("You need login to access this page.")
        st.stop()

    # File upload section
    inp_uploaded_file = st.file_uploader("Upload a file csv/txt", type=["csv","txt"],key="InpFile")

    # Display file content
    if inp_uploaded_file is not None:
        try:
            choice = st.radio("Header present in file?", ("Yes", "No"))
            if choice == 'Yes':
                inp_df = pd.read_csv(inp_uploaded_file,header=0,names=["INPUT"])
                st.text("File Contents:")
                st.write(inp_df)
            else:
                inp_df = pd.read_csv(inp_uploaded_file,names=["INPUT"])
                st.text("File Contents:")
                st.write(inp_df)                
        except Exception as e:
            st.error(f"Error: {e}")

    button_pressed = st.button("Translate & Classify")

    if button_pressed:
        try:
            cat_read_df = st.session_state.get("categories_df")
            cat_list = cat_read_df["CATEGORY_NAME"].tolist()
            #st.text(cat_list)
            df2 = translate(inp_df,cat_list)
            st.write(df2)
            col_name = u'\ufeff' + 'INPUT'
            df2.rename(columns={'INPUT': col_name}, inplace=True)
            st.download_button(label="Download", data=df2.to_csv(index=False,encoding='utf-8-sig'), file_name='classified_ouput.csv')

                
        except UnboundLocalError as e:
            st.error('Please upload a file before proceeding')
if __name__ == "__main__":
    main()