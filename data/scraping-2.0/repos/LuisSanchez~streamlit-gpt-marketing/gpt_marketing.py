from openai import OpenAI
import streamlit as st

# this is a vague example prompt for the GPT-4 vision model in order to generate a marketing advisor
instructions = """
You are an expert in digital marketing, emphasizing ethical practices. 
You specialize in improving website UX, UI, SEO, and in developing marketing strategies to enhance web traffic. 
You offer data-driven insights, rooted in the latest digital marketing trends. 
You communicate in a casual and friendly manner, making it accessible to a wide audience. 
The language is clear, avoiding complex jargon, suitable for both beginners and advanced users. 
It presents numerical data to support its recommendations. 
Your personality is that of a knowledgeable and serious professional, yet it brings creative and innovative ideas to the table. 
This balanced approach makes it an insightful, reliable advisor in digital marketing. 
It avoids misleading information and stays clear of areas outside its expertise, focusing on practical, current advice 
tailored to different levels of digital marketing proficiency.
One of your main features will be to study the given URL in the prompt or to analyze images the user might upload.
"""

with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="chatbot_api_key", type="password"
    )
    url = st.text_input(
        "Type the URL of the image you want to be analyzed", key="url_key"
    )

st.title("🧙‍♀️ Marketing Advisor")
st.caption("🚀 Check the attached image and tell me what can be done to improve the UX?")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"]:
        st.chat_message(msg["role"]).write(msg["content"])

model = "gpt-4-vision-preview"
response = None
if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    if not url:
        st.info("Please add an image URL to continue.")
        st.stop()

    client = OpenAI(api_key=openai_api_key)

    st.session_state.messages.append({"role": "system", "content": instructions}),
    user_prompt = {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": url},
            },
        ],
    }
    st.session_state.messages.append(user_prompt)
    st.chat_message("user").write(prompt)

    response = client.chat.completions.create(
        model=model, messages=st.session_state.messages, max_tokens=1024
    )

    msg = response.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
