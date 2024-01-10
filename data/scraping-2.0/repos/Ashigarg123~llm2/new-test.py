import openai 
import streamlit as st 
from streamlit_chat import message

openai.api_key = st.secrets["pass"] 

prg = st.progress(0)
def main_page():
    st.markdown("# Welcome to Machine Learning quiz! 😀")
    st.sidebar.markdown("# Machine learning quiz 🧐") 

def disable():
    st.session_state.disabled = True 

# Initialize disabled for form_submit_button to False
if "disabled" not in st.session_state:
    st.session_state.disabled = False

def page2():
    st.markdown("# Question 1❓")
    st.sidebar.markdown("# Machine learning quiz 🧐")
    

    #model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5",))
    
    with st.form("my_form"):
      st.write("Question 1 : Suppose a random variable X assumes the values -c and c with probabilities p and 1 -p respectively. What are the expectation E [X] and variance var(X)?")
      student_answer = st.text_area('Submit your answer below:')
      submitted1 = st.form_submit_button("Submit")
      if submitted1:
        if not student_answer.strip():
            st.error("WARNING: Please submit an answer")
        else:
            st.session_state.submit1 = True
            st.success("🎉🎉Your answer has been recorded!🎉🎉")
            st.balloons()
    st.markdown("Try, **ChatGPT** below:")
    ## chatgpt interface here
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    if 'model_name' not in st.session_state:
        st.session_state['model_name'] = []
    
    # if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo"
    # else:
    #     model = "gpt-4"
    clear_button = st.button("Clear Conversation", key="clear")
    # reset everything
    if clear_button:
        st.session_state['generated'] = []
        st.session_state['past'] = []
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        st.session_state['number_tokens'] = []
        st.session_state['model_name'] = []
    
    ## Container for chat history 
    response_container = st.container() 
    container = st.container()
    with container:
        with st.form(key='my_form2', clear_on_submit=True):
            user_input = st.text_area("Student:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')
        
        if submit_button and user_input:
            output= generate_response(user_input, model)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            st.session_state['model_name'].append("GPT-3.5")
    
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
                st.write(f"Model used: {st.session_state['model_name'][i]}")
           # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")

# function to generate a response
def generate_response(prompt, model):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    completion = openai.ChatCompletion.create(
         model=model,
         messages=st.session_state['messages']
     )
    response = completion.choices[0].message.content
    st.session_state['messages'].append({"role": "assistant", "content": response})
    return response


page_names_to_funcs = {
    "Main Page": main_page,
    "Question 1": page2,
}

selected_page = st.sidebar.selectbox("Select question number", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()