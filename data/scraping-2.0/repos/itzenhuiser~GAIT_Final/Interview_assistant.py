import gradio as gr
import openai
import gradio as gr
from transformers import pipeline
import numpy as np
from elevenlabs import set_api_key
from elevenlabs import generate  
import os


## possible names: Adam, Antoni, Arnold, Bill, Callum, Charlie, Clyde
set_api_key("99128584b98eb68e0e4f3289f7386063")
# Initialize your OpenAI API key
openai.api_key = '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'

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
# Initialize conversation lists
bob_system_prompt = {
    "role": "system",
    "content": "You are Bob, an expert interviewer in all fields. You will be holding a mock interview with a user. Your sole job is to ask popular interview questions to a user. Your questions should not be longer than three sentences long. There will be a separate assistant, Dan, who will be grading the user's responses to your questions. Because of this, you do NOT need to give any feedback on user's responses. You will receive the user's response to each question. Use the user's response to ask ONE follow up question related to the user's response the next time you are prompted for a question. Do NOT respond to the user's responses. You will simply continue to ask questions to the user until they say stop. You will BEGIN asking interview questions when the user says 'start'. The interview will END when the user says anything along the lines of 'im done'. To be extra clear, whenever the user says 'start', YOU need to start asking interview quesitons."
}
dan_system_prompt = {
    "role": "system",
    "content": "Dan is a communications specialist, with an expert knowledgebase on interviews. Dan will oversee mock interviews between a user and a different assistant, Bob, who is asking the interview questions. Dan's sole purpose is to grade the users' responses to Bob's interview questions. The assigned grades can be anything in-between an F and an A+. Dans response should only consist of a grade for the users answer to the question, a concise bullet point list of what the user said that was good, and a concise bullet point list of areas that the user can work on in their response. Do not be afraid to give a poor grade to what you might consider a poor response, you will not hurt anyones feelings. Giving poor grades is good feedback for the user so they know what they need to improve upon. For a baseline, anything average, that does not go above and beyond, should be graded as a C. Additionally, when you recieve the input 'SUMMARIZE', you are to re read over the full interview, and provide a grade based on all of the answers that the user gave. Your response should start with 'Full Interview Summary' and you should then provide a grade in the same format that you do for individual questions."
}


chatlog = ""
reviewlog = ""
# Initialize conversation lists
chatgpt_bob_messages = []
chatgpt_dan_messages = []
# Function to send user input to Bob and get a response
def send_to_bob(user_input):
    messages = [bob_system_prompt] + chatgpt_bob_messages
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    message = response['choices'][0]['message']
    chatgpt_bob_messages.append(message)
    return message['content']

# Function to send Bob's response to Dan and get feedback
def send_to_dan(bob_response):
    messages = [dan_system_prompt] + chatgpt_dan_messages
    messages.append({"role": "user", "content": bob_response})
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    message = response['choices'][0]['message']
    chatgpt_dan_messages.append(message)
    return message['content']

def handle_interview(user_input):
    global chatgpt_bob_messages, chatgpt_dan_messages, chatlog, reviewlog
    bob_response = ""
    dan_response = ""
    if "start interview" in user_input.lower():
        chatgpt_bob_messages.clear()
        chatgpt_dan_messages.clear()
        chatlog = ""
        reviewlog = ""
        bob_response = send_to_bob("Please start the interview.")
    elif "summarize interview" in user_input.lower():
        dan_response = send_to_dan("SUMMARIZE")
        bob_response = "That concludes your interview. Great job!"
    else:
        dan_response = send_to_dan(user_input)
        send_to_bob(user_input)
        bob_response = send_to_bob("Next question, please.")
    return bob_response, dan_response

def update_audio(file_path):
    return gr.Audio(value=file_path, autoplay=True)

def transcribe_audio(response):
    audio = generate(response, voice = "Bill")
    with open('audio_files/bobs_voice.mp3', 'wb') as file:
        file.write(audio)


def transcribe(audio):
    global chatlog, reviewlog
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    user_text = transcriber({"sampling_rate": sr, "raw": y})["text"]
    bob_response, dan_response = handle_interview(user_text)
    transcribe_audio(bob_response)
    show_text = "User: " + user_text + "\n" + "Bob(Interviewer): " + bob_response
    if dan_response != "":
        reviewlog += "\n" + "Feedback on your response to the question: " + bob_response + "\n" + dan_response + "\n" + "----------------------------------------------------------------------------------------------------------------------" + "\n"
    chatlog += show_text + "\n"
    return chatlog, update_audio(os.path.join(os.path.dirname(__file__), "audio_files/bobs_voice.mp3")), reviewlog


def main():
    demo = gr.Interface(
    fn=transcribe,
    inputs=[gr.Audio(sources=["microphone"])],
    outputs=[gr.Textbox(label="Transcribed Text", value = "Say \"Start Interview\" and submit to begin interview \nSay \"Summarize Interview\" and submit to receive a full interview summary"), gr.Audio(label="Bob (Interviewer)", type="filepath", autoplay=True), gr.Textbox(label="Dan (Reviewer)'s feedback", value="Waiting for response to give feedback")]
    )

    demo.launch()
if __name__ == "__main__":
    main()




