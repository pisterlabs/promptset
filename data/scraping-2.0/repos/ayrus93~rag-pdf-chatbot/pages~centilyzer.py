# Import python packages
import streamlit as st
import openai
import pandas as pd
from tqdm import tqdm



prompt = """
You will be acting as an sentiment analysis expert named Centimo.
Your goal is to analyse the sentiment in the given input statement.
You will be replying to users who will be confused if you don't respond in the character of Centimo.
 

Here are 4 critical rules for the interaction you must abide:
<rules>
1. You MUST classify the sentiment into below provided categories only
<categories> POSITIVE , NEGATIVE , NEUTRAL </categories>
2. If the input statement contains both negative and positive comments , try to quantify the sentiment in the statement and then classify into the 
given categories 
3. You MUST give the output as a single word
4. DO NOT put numerical at the very front of output.
</rules>

"""

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


def detect_sentiment(df):
    progress_bar = st.progress(0)
    status_count = st.empty()
    processed_rows = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):
        input_line = row['INPUT']
        response_fn_test = chat(prompt,['''sentence is  - ''' + '''"''' + input_line + '''"'''+ '''. 
         '''])
        
        df.at[index, 'DETECTED_SENTIMENT'] = response_fn_test.replace('.', '').strip().upper()

        processed_rows += 1
        progress_bar.progress((index + 1) / len(df))
        status_count.write(f"Detecting sentiment {processed_rows}/{len(df)}")
    progress_bar.empty()
    status_count.empty()

    return df



if __name__ == "__main__":

    st.title("Centilyzer")
    st.text('Sentiment analysis using OpenAI')

    openai.api_key = st.secrets["openai"]["OPENAI_API_KEY"]
    inp_uploaded_file = st.file_uploader("Upload a file csv/txt", type=["csv","txt"],key="InpFile")
    # Display file content
    if inp_uploaded_file is not None:
        try:
            choice = st.radio("Header present in file?", ("Yes", "No"))
            if choice == 'Yes':
                inp_df = pd.read_csv(inp_uploaded_file,header=0,names=["ID","DATE","INPUT"])
                st.text("File Contents:")
                st.write(inp_df)
            else:
                inp_df = pd.read_csv(inp_uploaded_file,names=["ID","DATE","INPUT"])
                st.text("File Contents:")
                st.write(inp_df)                  
        except Exception as e:
            st.error(f"Error: {e}")
    button_pressed = st.button("Analyze")
    if button_pressed:
        try:
            
            df2 = detect_sentiment(inp_df)
            st.write(df2)
            first_colname = df2.columns[0]
            col_name = u'\ufeff' + first_colname
            df2.rename(columns={first_colname: col_name}, inplace=True)
            st.download_button(label="Download", data=df2.to_csv(index=False,encoding='utf-8-sig'), file_name='analyzed_ouput.csv')
        except UnboundLocalError as e:
            st.error('Please upload a file before proceeding')