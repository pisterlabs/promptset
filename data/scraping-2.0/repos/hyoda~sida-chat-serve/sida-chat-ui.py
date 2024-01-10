import openai
import streamlit as st

from streamlit_chat import message

openai.api_key = st.secrets['pass']

# def generate_response(propt):
#     try:
#       completion = openai.Completion.create(
#           engine = "text-davinci-003",
#           prompt = propt,
#           max_tokens = 1024,
#           n = 1,
#           stop = None,
#           temperature = 0.5,
#       )
#       message = completion["choices"][0]["text"]
#       return message
#     except openai.error.OpenAIError as e:
#       st.error(f"An error occurred: {e}")


# st.title("시다 챗: 남해워케이션에서 만든 인공지능 챗봇")


# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []


# if 'generated' not in st.session_state:
#     st.session_state.generated = []

# if 'past' not in st.session_state:
#     st.session_state.past = []

# def get_text():
#     input_text = st.text_input("나: ", "", key="input")
#     return input_text

  

# if st.session_state.generated:
#     # st.write("Sida: ", st.session_state.generated[-1])
#     for i in range(len(st.session_state['generated']) -1, -1, -1):
#         message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
#         message(st.session_state['generated'][i], key=str(i))

# # 사용자 정의 스타일을 포함한 HTML 및 CSS
# custom_style = """
# <style>
# .stTextInput {
#     position: fixed;
#     top: 10px;
#     width: 95%;
#     z-index: 9999;
# }
# .stText {
#     margin-top: 40px;
# }
# </style>
# """

# # Streamlit의 experimental_set_query_params 함수를 사용하여 커스텀 스타일을 적용
# st.experimental_set_query_params(__st_custom=custom_style)
      


# user_input = get_text()

# col1, col2 = st.columns(2)

# if user_input:
#     st.session_state.chat_history.append(("User", user_input))
#     bot_response = generate_response("\n".join([item[1] for item in st.session_state.chat_history]))
#     st.session_state.chat_history.append(("Sida", bot_response))

# for role, text in st.session_state.chat_history:
#     if role == "User":
#         message(text, is_user=True)
#     else:
#         message(text)



st.title("시다 챗: 남해워케이션에서 만든 인공지능 챗봇")


# Initialise session state variables
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
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Sidebar")
model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Map model names to OpenAI model IDs
if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo"
else:
    model = "gpt-4"

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


# generate a response
def generate_response(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})

    completion = openai.ChatCompletion.create(
        model=model,
        messages=st.session_state['messages']
    )
    response = completion.choices[0].message.content
    st.session_state['messages'].append({"role": "assistant", "content": response})

    # print(st.session_state['messages'])
    total_tokens = completion.usage.total_tokens
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    return response, total_tokens, prompt_tokens, completion_tokens


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output, total_tokens, prompt_tokens, completion_tokens = generate_response(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        st.session_state['model_name'].append(model_name)
        st.session_state['total_tokens'].append(total_tokens)

        # from https://openai.com/pricing#language-models
        if model_name == "GPT-3.5":
            cost = total_tokens * 0.002 / 1000
        else:
            cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

        st.session_state['cost'].append(cost)
        st.session_state['total_cost'] += cost

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            st.write(
                f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
            counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")