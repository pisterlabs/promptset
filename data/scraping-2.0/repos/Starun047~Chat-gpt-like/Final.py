import openai
import streamlit as st
from streamlit_pills import pills

openai.api_key = st.secrets['api_secret']

st.subheader("AI Assistant : Streamlit + OpenAI: `stream` *argument*")
selected = pills("", ["NO Streaming", "Streaming"], ["🎈", "🌈"])

user_input = st.text_input("You: ",placeholder = "Ask me anything ...", key="input")


if st.button("Submit", type="primary"):
    st.markdown("----")
    res_box = st.empty()
    if selected == "Streaming":
        report = []
        for resp in openai.Completion.create(model='text-davinci-003',
                                            prompt=user_input,
                                            max_tokens=120, 
                                            temperature = 0.5,
                                            stream = True):
            # join method to concatenate the elements of the list 
            # into a single string, 
            # then strip out any empty strings
            report.append(resp.choices[0].text)
            result = "".join(report).strip()
            result = result.replace("\n", "")        
            res_box.markdown(f'*{result}*') 
            
    else:
        completions = openai.Completion.create(model='text-davinci-003',
                                            prompt=user_input,
                                            max_tokens=120, 
                                            temperature = 0.5,
                                            stream = False)
        result = completions.choices[0].text
        
        res_box.write(result)
st.markdown("----")