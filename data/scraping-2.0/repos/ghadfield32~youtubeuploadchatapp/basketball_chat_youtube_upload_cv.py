import cv2
import streamlit as st
import openai
from pytube import YouTube
from ultralytics import YOLO


def chatbot_ui(message=None):
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = ""

    if message:
        st.session_state.chat_log += f"🏀 **Pro says:** {message}\n"

    prompt = st.sidebar.text_input("Your message:")

    if prompt:
        st.session_state.chat_log += f"**You:** {prompt}\n"
        try:
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
            response_text = response.choices[0].message["content"].strip()
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.session_state.chat_log += f"🏀 **Pro says:** {response_text}\n"
        except openai.error.RateLimitError:
            st.session_state.chat_log += "Sorry, seems like the Pro is taking a break. Please try again later!\n"

    st.sidebar.markdown(st.session_state.chat_log)



def main():
    st.set_page_config(
        page_title="Basketball Pro's Video Analysis",
        layout="wide",
        initial_sidebar_state="auto",
    )
    
    openai.api_key = st.secrets["openai_key"]
   
    st.sidebar.title("🏀 Chat with a Basketball Pro!")
    st.sidebar.write("Hey there! I've seen plenty of plays in my time. Share a video and let me guide you through what I see.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    link = st.sidebar.text_input("Enter YouTube Video Link:")

    if link:
        chatbot_ui("Alright, rookie! Give me a sec to download and analyze the video. Hold on to your basketballs!")

        yt = YouTube(link)
        stream = yt.streams.filter(file_extension="mp4").first()
        video_path = stream.download()

        model = YOLO('yolov8n.pt')

        chatbot_ui("Let's break down this video play-by-play. Watch closely!")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('processed_video.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
        frame_placeholder = st.empty()

        ret = True
        while ret:
            ret, frame = cap.read()
            if ret:
                results = model.track(frame, persist=True)
                frame_ = results[0].plot()
                out.write(frame_)
                frame_rgb = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels='RGB', use_column_width=True)
                cv2.waitKey(delay)

        cap.release()
        out.release()

        chatbot_ui("And there you have it! Feel free to ask if you have questions about the breakdown.")

    else:
        chatbot_ui()


if __name__ == "__main__":
    main()
