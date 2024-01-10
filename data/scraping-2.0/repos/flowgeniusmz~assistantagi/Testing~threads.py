import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=st.secrets.openai.api_key_general)

def create_thread():
    thread = client.beta.threads.create()
    threadid = thread.id
    return threadid

def add_initial_msgs_to_thread(vMessages):
    thread_messages = []
    threadid = create_thread()
    for msg in vMessages:
        thread_message = client.beta.threads.messages.create(
            thread_id=threadid,
            role="user",
            content=msg
        )
        thread_message_id = thread_message.id
        thread_messages.append(thread_message_id)
    return threadid, thread_messages


weather_thread_msgs = [
    "Weather forecast for today is sunny with a slight chance of rain in the evening.",
    "Did you know about the El Niño weather phenomenon? It can drastically affect climate patterns.",
    "Always carry an umbrella in unpredictable weather. Better safe than sorry!"
]

coding_thread_msgs = [
    "Remember, writing clean, readable code is as important as making it work.",
    "Debugging tip: Always check for off-by-one errors in your loops.",
    "Code reviews are a great way to improve your programming skills."
]

travel_thread_msgs = [
    "Travel tip: Always research local customs and etiquette before visiting a new place.",
    "Top 5 destinations to visit this year: Japan, New Zealand, Italy, Canada, and Brazil.",
    "Packing light can make your travel more comfortable and flexible."
]

threadid_weather, threadmessages_weather = add_initial_msgs_to_thread(weather_thread_msgs)
threadid_coding, threadmessages_coding = add_initial_msgs_to_thread(coding_thread_msgs)
threadid_travel, threadmessages_travel = add_initial_msgs_to_thread(travel_thread_msgs)

print(threadid_weather)
print(threadmessages_weather)
print(threadid_coding)
print(threadmessages_coding)
print(threadid_travel)
print(threadmessages_travel)

#thread_hraF0iFGXmy09qhKnJSrlr7J
#['msg_8ey8jorzrN0vtZeigghWEprE', 'msg_wr7rXP39hjmV3Tax94ZchW2j', 'msg_0TdKEy3b2T5JAMuAHsJI5H6i']
#thread_x6WnC2qNpz2Mp1FxUwu0aI4i
#['msg_b2vXzYLuLEN84f3WvXMoi35m', 'msg_kyckSPGQPM2yV2EdczPWFWPD', 'msg_C284L7D6Vba3zdG3YcMtht0E']
#thread_o3l2RRco00gQUiXmUMaXuDlI
#['msg_6vZYcx9XaKfOo3w9TggsEQyH', 'msg_4gAwIDXiAyVOpLSfSKgnYEKF', 'msg_nZQE5HNn9YIR3PddkJ51AgEN']
