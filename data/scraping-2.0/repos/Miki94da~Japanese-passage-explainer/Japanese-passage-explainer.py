
import streamlit as st
import openai
import json
import pandas as pd
import json.decoder



### streamlit structure
st.title('Japanese passage :red[explainer]')
st.subheader('Input paragraph that you want to understand its vocabularies. The AI will explian you!!!!.', divider='red')
lang_option = st.selectbox("Please choose language you want the text to be analysed \n ", ("Easy Japanese", "English", "Thai"))

user_input = st.text_area("Your Japanese text :", )



# Get the API key from the sidebar called OpenAI API key
user_api_key = st.sidebar.text_input("OpenAI API key", type="password")

client = openai.OpenAI(api_key=user_api_key)
prompt_vocab = r"""Act as an AI tutor in Japanese. You will receive a japanese passage and you must select maximum 15 interesting vocabulary excluding specific nouns to explain in each field below.
List each vocabulary in a JSON array(list of dictionary).
Each suggestion should have these fields:
"vocab" - the vocab in plain form and put hiragana pronunciation in ()
"meaning"- the meaning of the vocab in {lang_option}
"part of speech" - part of speech of that vocab for example Noun,Verb,adj-i ,adj-na,adverb
"example"- the example usage of the vocab
"synonym" - the synonym(類語) of the vocab
Don't say anything at first. Wait for the user to say something.
Make sure it's a valid JSON array with the fields above. 
if there id no synonym, put "No synonym" in the field.
"""

prompt_sum = r"""Act as an professional in Japanese.
You will receive a japanese passage and you should summarize it into {lang_option} language.
must translate the passage into {lang_option} language.
"""

max_attempts = 5
attempt_count = 0
success = False
               
if lang_option == "Easy Japanese":
        prompt_sum = prompt_sum.replace("{lang_option}", "Easy Japanese")
        prompt_vocab = prompt_vocab.replace("{lang_option}", "Easy Japanese")
elif lang_option == "English":
        prompt_sum = prompt_sum.replace("{lang_option}", "english")
        prompt_vocab = prompt_vocab.replace("{lang_option}", "english")
elif lang_option == "Thai":
        prompt_sum = prompt_sum.replace("{lang_option}", "Thai")
        prompt_vocab = prompt_vocab.replace("{lang_option}", "Thai")
user_input = user_input.replace(' "\'" ', "\"")

submit = st.button('Submit')
# submit button after text input
while submit and attempt_count < max_attempts and not success: ## new
   

    messages_so_far_vocab = [
        {"role": "system", "content": prompt_vocab},
        {'role': 'user', 'content': user_input},
    ]
    response_vocab = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages_so_far_vocab
    )

    message_sum = [
        {"role": "system", "content": prompt_sum},
        {'role': 'user', 'content': user_input},
    ]
    response_sum = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=message_sum
    )
    response_sound = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input= user_input ,
        ) 
    response_sound.stream_to_file("output.mp3")
    audio_file = open("output.mp3", 'rb')
    audio_bytes = audio_file.read()
    sample_rate = 10000
    

    try:
        suggestion_dictionary_vocab = response_vocab.choices[0].message.content
        suggestion_dictionary_sum = response_sum.choices[0].message.content
        print(suggestion_dictionary_vocab)
        print(suggestion_dictionary_sum)
        sd_vocab = json.loads(suggestion_dictionary_vocab, strict=False)
        suggestion_df = pd.DataFrame.from_dict(sd_vocab) 
        if  suggestion_df.shape[1] == 5 :
            st.write('let\'s listen to the sound :')
            st.audio(audio_bytes, start_time=-1)
            st.markdown("Summarized text :")
            st.write(suggestion_dictionary_sum)
            st.markdown("Vocabularies :")
            st.dataframe(sd_vocab)
            st.balloons()
            success = True

    except (json.decoder.JSONDecodeError, ValueError) as E:
        attempt_count += 1

if attempt_count == max_attempts:
    st.error('Please submit again or change the text.')
