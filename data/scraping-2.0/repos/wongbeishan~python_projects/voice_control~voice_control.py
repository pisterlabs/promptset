import whisper
import openai
import gradio as gr
import librosa
from dotenv import load_dotenv


load_dotenv()
model = whisper.load_model("base")


def load_audio(file_path, target_sr = 1000):
    try:
        audio = librosa.load(file_path, sr=target_sr)
        if len(audio.shape) > 1:
            audio = audio[0]

        return audio
    
    except Exception as e:
        print("Error loading audio file: {e}")
        return None


def transcribe(file):
    # print(file)
    audio = load_audio(file)
    if audio is None:
        print("Error loading audio file.")
        return "Error loading audio file."
    transcription = model.transcribe(file)
    return transcription


def generate_answer(messages):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message.content


prompts = {
    "START": "Classify the intent of the next input. \
            Is it: WRITE_EMAIL, QUESTION, OTHER ? Only answer one word.", "QUESTION": "If you can answer the question: ANSWER, \
            if you need more information: MORE, \
            if you cannot answer: OTHER. Only answer one word.",
    "ANSWER": "Now answer the question",
    "MORE": "Now ask for more information",
    "OTHER": "Now tell me you cannot answer the question or do the action", "WRITE_EMAIL": 'If the subject or recipient or message is missing, \
            answer "MORE". Else if you have all the information, \ answer "ACTION_WRITE_EMAIL |\
            subject:subject, recipient:recipient, message:message".',
}

actions = {
    "ACTION_WRITE_EMAIL": "The mail has been sent. \
    Now tell me the action is done in natural language."
}


def start(user_input):
    messages = [{"role": "user", "content": prompts["START"]}]
    messages.append({"role": "user", "content": user_input})
    return discussion(messages, "")


def discussion(messages, last_step):
    answer = generate_answer(messages)
    if answer in prompts.keys():
        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": prompts[answer]})
        return discussion(messages, answer)
    elif answer in actions.keys():
        do_action(answer)
    else:
        if last_step != "MORE":
            messages = []
        last_step = "END"
        return answer


def do_action(action):
    print("Doing action " + action)
    return ("I did the action " + action)


def start_chat(file):
    input = transcribe(file)
    return start(input)


gr.Interface(
    fn=start_chat,
    live=True,
    inputs=gr.Audio(sources="microphone"),
    outputs="text"
).launch()


