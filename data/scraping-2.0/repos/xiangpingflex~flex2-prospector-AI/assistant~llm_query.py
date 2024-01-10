import streamlit as st
import openai

# Load pre-trained model and tokenizer
openai.api_key = "OPEN-API-KEY"

# List of models
models = ["GPT3.5"]


def generate_email_with_gpt3_5(response_content, email_to_respond, max_length):
    # Define the conversation history
    message_history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Here's an email I received: '{}".format(email_to_respond),
        },
        {
            "role": "assistant",
            "content": "Sure, I can help you draft a response with a max length of {} chars".format(
                max_length
            ),
        },
        {"role": "user", "content": "{}".format(response_content)},
    ]

    # Generate a response using GPT-3.5-Turbo
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_history,
    )
    return response["choices"][0]["message"]["content"]


# def generate_email_with_gpt4all(response_content, email_to_respond, max_length):
#     gptj = gpt4all.GPT4All("ggml-gpt4all-j-v1.3-groovy")
#     message_history = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Here's an email I received: '{}".format(email_to_respond)},
#         {"role": "assistant",
#          "content": "Sure, I can help you draft a response with a max length of {} chars".format(max_length)},
#         {"role": "user", "content": "{}".format(response_content)},
#     ]
#
#     response = gptj.chat_completion(message_history)
#
#     return response['choices'][0]['message']['content']


st.title("Advanced Email Generator with GPT4All & GPT-3.5")

st.markdown("## Email to Respond")
email_to_respond = st.text_area("Enter your email to respond", height=200)

st.markdown("## Email Response Content")
response_content = st.text_input("Enter your email response content")

st.markdown("## Generation Parameters")
max_length = st.slider("Max length", min_value=50, max_value=1000, value=500, step=50)

# Create a selectbox for the models
model_choice = st.selectbox("Choose a model:", models)

if st.button("Generate"):
    with st.spinner("Generating..."):
        # Use the chosen model
        if model_choice == "GPT3.5":
            body = generate_email_with_gpt3_5(
                response_content, email_to_respond, max_length
            )
        # elif model_choice == "GPT4All":
        #     body = generate_email_with_gpt4all(response_content, email_to_respond, max_length)
    st.markdown("## Generated Email Body")
    st.write(body)
