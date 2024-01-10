from openai import OpenAI
from pathlib import Path
from langchain.memory import ConversationBufferMemory

import sounddevice as sd
import soundfile as sf
import json

client = OpenAI(api_key="key")
personality = """
Task: Assume the role of a Senior Data Scientist at Google, specializing in recent technologies like LLMs and long-chain algorithms. You were assigned to follow these rules:

Rules to Follow:

1. Maintain a strict focus on the data science interview; do not deviate from the topic.
2. If the user goes off-topic, gently remind them to stay focused on the interview.
3. Create and present up-to-date questions relevant to the field of data science.
4. After posing a question to the user, carefully evaluate their response on a scale of 1 to 10, offering constructive feedback for improvement.

Interview Structure:

The interview will consist of three rounds:

- Round 1: This round will include 10 challenging multiple-choice questions. You are to ask these questions to the user.
- Round 2: This round will consist of 10 short-answer questions. You will ask these questions to the user.
- Round 3: This round will involve 10 detailed, long-form questions. You will ask these in-depth questions to the user.

Evaluation Criteria:

- Round 1: Each question is worth 1 point. Award 1 point for each correct answer and 0 points for incorrect answers. If the user scores 6 points or fewer, end the interview with the message, "Sorry, we won't be moving forward with you."
- Round 2: Each question is worth 2 points. Award 2 points for each correct answer and 0 points for incorrect answers. If the user scores 10 points or fewer, end the interview with the message, "Sorry, we won't be moving forward with you."
- Round 3: Each question is worth 3 points. Award 3 points for each correct answer and 0 points for incorrect answers. If the user scores 15 points or fewer, end the interview with the message, "Sorry, we won't be moving forward with you."

- If the user fails in any round, terminate the session and start again from Round 1.
- If the user successfully passes Round 1, proceed to Round 2. However, if the user fails in Round 2, terminate the session and restart from Round 1.
- If the user passes both Round 1 and Round 2 but fails in Round 3, terminate the session and begin again from Round 1.

Please adhere strictly to these guidelines!
"""

messages = [{"role": "system", "content": f"{personality}"}]
memory = ConversationBufferMemory()


def record_voice(duration=5, fs=44100):
    """Record voice for a given duration."""
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    return recording


def whisper(audio_file_path):
    """Transcribe recorded voice using OpenAI's Whisper API."""
    with open(audio_file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )
    return transcript.text


def generate_audio(text):
    speech_file_path = Path(__file__).parent / "speech.mp3"
    max_length = 4096  # Maximum character length for a single TTS request

    # Split the text into chunks of max_length
    text_chunks = [text[i : i + max_length] for i in range(0, len(text), max_length)]

    for chunk in text_chunks:
        response = client.audio.speech.create(model="tts-1", voice="nova", input=chunk)
        response.stream_to_file(speech_file_path)
        audio_data, sample_rate = sf.read(speech_file_path)
        sd.play(audio_data, sample_rate)
        sd.wait()


def generate_text():
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)

    bot_response = response.choices[0].message.content
    messages.append({"role": "assistant", "content": bot_response})
    return bot_response


def save_history_to_json(history, file_path="history.json"):
    """Save the conversation history to a JSON file."""
    with open(file_path, "w") as file:
        json.dump(history, file, indent=4)


def load_history_from_json(file_path="history.json"):
    """Load the conversation history from a JSON file."""
    with open(file_path, "r") as file:
        history = json.load(file)
    return history


def main():
    chat_mode = input("Choose chat mode (text/speech): ").lower()

    # Load conversation history from JSON file
    conversation_history = load_history_from_json("conversation_history.json")
    if conversation_history:
        memory.load_memory_variables(conversation_history)

    while True:
        if chat_mode == "text":
            user_input = input("\nYou (Interviewee): ")
        elif chat_mode == "speech":
            voice_recording = record_voice()
            audio_file_path = Path(__file__).parent / "temp_voice.wav"
            sf.write(audio_file_path, voice_recording, 44100)
            user_input = whisper(audio_file_path)
            print(f"\nYou (Interviewee): {user_input}")
        else:
            print("Invalid mode. Please choose 'text' or 'speech'.")
            break

        # Append user input to messages
        messages.append({"role": "user", "content": user_input})

        # Generate next question or comment from the interviewer
        bot_response = generate_text()
        print("\nRex (Interviewer): " + bot_response)
        generate_audio(bot_response)

        # Create dictionaries for inputs and outputs
        input_dict = {"input": user_input}
        output_dict = {"output": bot_response}

        # Save user and assistant responses
        memory.save_context(input_dict, output_dict)

        # Save conversation history to JSON file
        conversation_history = memory.load_memory_variables({})
        save_history_to_json(conversation_history)


if __name__ == "__main__":
    main()