import streamlit as st
from deep_translator import GoogleTranslator
from pytube import YouTube
import random
import base64
import os
from langchain.llms import OpenAI
from langchain.agents import AgentExecutor, AgentType, initialize_agent, load_tools  # type: ignore
from langchain.tools import BaseTool
from typing import List
import openai

messages = [
    {"role": "system", "content": "Investor seeking guidance. Provide a credibility score (1-100) for the given content, considering source reliability (40%), market conditions (30%), and risk factors (30%). Your response format: credibility score: (your answer) in one line, followed by reason:  in a concise paragraph (max 150 words). Emphasize due diligence importance, exercise caution, and maintain a highly critical approach. Address fraudulent activities and refrain from accepting information without proper evidence. The user relies on your assessment for investment decisions, so precision is crucial. The content is as follows:{topic} "},
]


def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("image.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://media.istockphoto.com/id/1252292286/photo/black-bull-and-bear.jpg?s=170667a&w=0&k=20&c=EGv51h_SIWTHlaLYnxr6TfM5pKsv6nvudjW1IrYIeS8=");
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


def main():
    st.title("FraudShield")
    tabs = st.tabs(["Home", "Text Input", "Image Input", "YouTube URLs"])

    with tabs[0]:
        st.subheader(
            "Welcome to FraudShield - Your Trusted Source for Financial Safety"
        )
        st.write(
            "At FraudShield, we're committed to safeguarding your financial well-being in the digital age. We understand that in today's fast-paced world, it's increasingly challenging to separate fact from fiction when it comes to financial news and investment opportunities. That's why we're here to help you make informed decisions and protect your hard-earned money."
        )

    with tabs[1]:
        user_input_key = "userinput"
        userinput = st.text_input("Enter your statement here:", key=user_input_key)

        if st.button("Verify"):
            with st.spinner("Analyzing the statement"):
                # Append user's message to messages
                messages.append({"role": "user", "content": userinput})

                # Call OpenAI's Chat API
                chat = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k-0613",
                    messages=messages,
                )

                # Get the assistant's reply
                userinput = chat.choices[0].message["content"]

                # Append assistant's reply to messages
                messages.append({"role": "assistant", "content": userinput})
            user_input1=userinput[:22]
            user_input2=userinput[22:]
            st.write(user_input1)
            st.write(user_input2)
            # Display the assistant's reply
            # st.write(userinput)

    with tabs[3]:
        
        user_input_key = "user_input"
        user_input = st.text_input("Paste the link here:", key=user_input_key)

        openai.api_key =st.secrets["OPENAI_API_KEY"]

        if st.button("Check"):
            videoReview(user_input)

    with tabs[2]:
        st.subheader("Image")

        image = st.file_uploader(
            label="Upload an image",
            type=["jpg", "png"],
            accept_multiple_files=False,
        )

        if st.button("Submit"):
            if image is None:
                st.error("Please upload an image.")
            else:
                # save the image
                with st.spinner("Saving image..."):
                    image_path = "./temp.jpg"
                    with open(image_path, "wb+") as f:
                        f.write(image.getbuffer())

                # reading text from image
                with st.spinner("Extracting text..."):
                    import easyocr

                    Reader = easyocr.Reader(["en"])
                    text = " ".join(Reader.readtext(image_path, detail=0))
                    res=fraud(text).strip().capitalize()
                    # res = fraud(text).strip().capitalize()
                    # if res.find("s") == 0 or res.find("S") == 0:
                    #     st.error(res)
                    # else:
                    #     st.success(res)
                    res1=res[:22]
                    res2=res[22:]
                    st.write(res1)
                    st.write(res2)

                # delete image
                with st.spinner("Cleaning up..."):
                    os.remove(image_path)


def fraud(search: str) -> str:
    # Create a new instance of the OpenAI class
    llm = OpenAI(
        openai_api_key=st.secrets["OPENAI_API_KEY"],
        max_tokens=200,
        temperature=0,
        client=None,
        model="text-davinci-003",
        frequency_penalty=1,
        presence_penalty=0,
        top_p=1,
    )

    # Load the tools
    tools: List[BaseTool] = load_tools(["google-serper"], llm=llm)

    # Create a new instance of the AgentExecutor class
    agent: AgentExecutor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )

    template = """Investor seeking guidance. Provide a credibility score (1-100) for the given content, considering source reliability (40%), market conditions (30%), and risk factors (30%). Your response format: 'credibility score: (your answer)' in one line, followed by 'reason: ' in a concise paragraph (max 150 words). Emphasize due diligence importance, exercise caution, and maintain a highly critical approach. Address fraudulent activities and refrain from accepting information without proper evidence. The user relies on your assessment for investment decisions, so precision is crucial.The content is as follows: {topic}"""

    # Generate the response
    response: str = agent.run(template.format(topic=search))

    # Print the response
    print(response)

    # # Convert the response to a dictionary
    # result = json.loads(response)

    return response
def videoReview(yt_link):
    video_caller = YouTube(yt_link)
    a = random.randint(1, 10000)
    a = str(a)
    titlename=video_caller.title
    video_caller.title = a
    with st.spinner("Downloading the video"):
        video_caller.streams.filter(only_audio=True).first().download()
    b = a + ".mp4"
    with st.spinner("Extracting the content"):
        with open(b, "rb") as audio_file:
            transcript2 = openai.Audio.transcribe(
                file=audio_file, model="whisper-1", response_format="srt", language="en"
            )
    # st.write(transcript2)
    if transcript2:
        # Append user's message to messages
        messages.append(
            {
                "role": "user",
                # "content": transcript2,
                "content": "Video title name : "+titlename+"\n"+"transcription: "+transcript2,
            })
        
        # print(messages)
        # st.write(messages)
        with st.spinner("Analyzing the content"):
        # Call OpenAI's Chat API
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k-0613",
                messages=messages,
            )

        # Get the assistant's reply
        user_input = chat.choices[0].message["content"]

        # Append assistant's reply to messages
        # messages.append({"role": "assistant", "content": user_input})

        # Display the assistant's reply
        userinput1=user_input[:22]
        userinput2=user_input[22:]
        st.write(userinput1)
        st.write(userinput2)


def translate(text: str) -> str:
    translator: GoogleTranslator = GoogleTranslator(source="auto", target="en")
    text = translator.translate(text)
    text = (
        text.strip()
        .replace("\n", " ")
        .replace("\t", " ")
        .replace("\r", " ")
        .rstrip(".")
    )
    return text


if __name__ == "__main__":
    main()