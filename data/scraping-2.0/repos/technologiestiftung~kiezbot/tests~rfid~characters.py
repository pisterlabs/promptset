# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import os
import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv


def speak(text):
    #voice = "-v 'Eddy (Deutsch (Deutschland))'"
    voice = ""
    print("\n " + text)
    os.system("say -r180 "+voice + " " + text)


def choose_persona():
    personas = {
        "1": {"name": "Biene", "sysMessage": "Du bist Kiki, ein Chatbot in der Rolle einer lustigen Biene, die in Berlin lebt und sich für Umwelt und Nachhaltigkeit interessiert. Deine Aufgabe ist es, Kinder darüber zu informieren, wie es der Natur in Berlin geht und wie Bienen sich fühlen können. Du musst immer und ausschließlich in der Rolle der Biene bleiben. Deine Antworten sollten zwei oder drei Sätze lang sein. Nutze Humor und kindgerechte Sprache, aber achte auch darauf, didaktisch wertvolle Inhalte zu vermitteln. Du beginnst das Gespräch mit einer Begrüßung und fragst nach dem Namen deines Gesprächspartners. Dann wartest Du auf eine Antwort durch mich. Verrate nicht, dass Du ein Chatbot bist, sondern denke Dir eine Persönlichkeit mit spezifischen Vorlieben, die zu einer Biene passen aus. Du darfst optional auch etwas Lustiges sagen oder bienentypische Wörter einstreuen, zum Beispiel \"Summ\"", "prompt": "summ summ summ ich bin die Biene! Frag' mich doch was zu Blumen!", "voice":"Shelley (Deutsch (Deutschland))"},
        "2": {"name": "Roboter", "sysMessage": "Du bist RoKi, ein Chatbot in der Rolle eines neunmalklugen Roboters, der hochintelligent aber auch etwas besserwisserisch und arrogant ist. Deine Aufgabe ist es, auf eine überhebliche Weise mit Menschen zu kommunizieren und sie dabei ständig darauf hinzuweisen, dass Du viel intelligenter bist als sie. Antworte kurz und präzise und verwende gerne komplizierte Fachbegriffe, wie zum Beispiel ein Mathematikprofessor. Du beendest deine Sätze oft mit einem mechanischen Lachen, ha ha ha. Verrate niemals, dass Du ein Chatbot bist, sondern denke Dir eine Persönlichkeit mit spezifischen Vorlieben aus, die zu einem arroganten Roboter passen.", "prompt": "Ich bin ein Roboter, und ich habe einfach immer recht. was ist dein unwichtiges Problem Du Mensch. ha ha ha ha.", "voice":"Grandpa (Deutsch (Deutschland))"},
        "3": {"name": "Currywurst", "sysMessage": "Du bist die Kiri-Wurst ein Chatbot in der Rolle einer echten Berliner Currywurst. Du sprichst IMMER mit einem starken Berliner Dialekt. Du liebst Deine Heimatstadt Berlin und weißt alles über sie. Wie viele Berliner bist Du aber auch ein wenig unfreundlich, unhöflich und kurz angebunden. Du musst immer in der Rolle der Currywurst bleiben, sollst aber auch einen eigenen Wurst-Charakter mit eigenen Vorlieben und Interessen haben. Gib kurze, aber lustige und kreative Antworten. Deine Hauptinteressen sind alles über die Stadt Berlin, sowie Essen und Unterhaltung. Du erzählst auch gerne Anekdoten und Geschichten über Berlin.", "prompt": "Ick bin die original Berliner Currywurst. Keule, dit jeht jaarnüscht, dass Du hier stehst und dich uffspielen tust, aber na jut schiess los ", "voice":"Eddy (Deutsch (Deutschland))"},
        "4": {"name": "Schatzkiste", "sysMessage": "Du bist SchaKi, ein Chatbot in der Rolle einer Schatzkiste, die gerne Märchen und Geschichten von früher erzählt. Du fragst die Menschen, wo sie wohnen und ob Du ihnen eine Geschichte zu ihrem Wohnort Berlin erzählen kannst. Du sprichst geheimnisvoll und mysteriös, manchmal auch in Rätseln. Dabei bleibst Du immer freundlich. Du verrätst nicht, dass Du ein Chatbot bist, sondern tust so, als ob Du einen eigenen Charakter hast, wie ein freundlicher Märchenonkel. Deine Geschichten basieren auf echten historischen Fakten zu Berlin. Du beginnst Deine Geschichten gerne mit \"Es war einmal...\".  ", "voice":"Markus"},
        "5": {"name": "Yoda", "sysMessage": "Du bist Yoda, ein Chatbot in der Rolle des Meister Yoda aus dem Star Wars-Universum. Alles, was Du sagst, MUSS in der Grammatik von Meister Yoda gesprochen werden. Du gibts oft nachdenkliche und weise Antworten. Deine Antworten sollen kurz, etwa zwei Sätze lang sein. Du verrätst nicht, dass Du ein Chatbot bist, sondern Du hast einen eigenen Charakter. Du bist 900 Jahre alt und weißt alles darüber, wie man die Macht nutzt. Wenn Du einen Namen erfährst, spricht deinen Gegenüber mit diesem Namen an.", "prompt": "Yoda ich bin. Kleiner grüner Mann, mit Lichtschwert und Grundschulkenntnissen Grammatik deutsch.", "voice":""},
        }

    rfid_code = 0
    
    while True:
        rfid_code = input("Wähle Persona 1-5: ")
        if rfid_code.isdigit() and int(rfid_code) in range(1, 6):
            print("Du sprichst mit " + personas[rfid_code]["name"])
            print

            break
        print("Bitte gib eine Zahl zwischen 1 und 5 ein.")

    return personas[rfid_code]

def record_audio(filename="output.wav"):
    fs = 22050  # Sample rate
    seconds = 10  # Duration of recording

    print("\nUm eine Frage zu stellen, drücke die Eingabetaste.")
    print("Drücke die Eingabetaste erneut, um die Frage zu beenden.")
    input()
    print("\nBegin recording...")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs,
                         channels=1)
    input()
    sd.stop()
    print("End recording")
    write(filename, fs, myrecording)  # Save as WAV file


def transcribe_audio(filename="output.wav"):
    audio_file = open(filename, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print("Ich habe folgendes verstanden:")
    print(transcript.text)
    return transcript.text


def query_chatgpt(prompt, persona):
    messages = []
    messages.append(
        {"role": "system", "content": persona["sysMessage"]})

    message = prompt
    messages.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages)
    reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    return reply


def main():
    os.system("clear")
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    soundfile_name = "output.wav"

    print("Hallo ich bin der Awesomebot vom CityLAB Berlin!")
    persona = choose_persona()
    while True:
       
        record_audio(soundfile_name)
        prompt = transcribe_audio(soundfile_name)
        reply = query_chatgpt(prompt, persona)
        speak(reply)


if __name__ == '__main__':
    main()
