import pyttsx3

# Initialisiere den Text-to-Speech-Engine
engine = pyttsx3.init()

# Ändere die Stimme zur "Microsoft Hedda"-Stimme
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id) #einzige deutsche Stimme

text="Hallo, mein Name ist Heymex. Darf ich ihre Sauerstoffsättigung messen?"
# Konvertiere Text in Sprache
engine.say(text)

# Convert text to speech and save it as an MP3 file
engine.save_to_file(text, 'output.mp3')

# Warte, bis die Konvertierung abgeschlossen ist
engine.runAndWait()



# NOTE: this example requires PyAudio because it uses the Microphone class

# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai


# Definieren Sie den Text, auf den geantwortet werden soll
openai.api_key = ''
audio_file= open("Aufnahme.m4a", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)
text1=str(transcript)
text1=text1[12:]
print(text1)
"""# Verwende die create()-Methode, um eine Antwort von der KI zu erhalten
response = openai.Completion.create(
  engine="ada", # wähle eines der verfügbaren GPT-3-Modelle
  prompt=str(text1),
  max_tokens=60 # Anzahl der Token in der Antwort
)

# Extrahiere die Antwort aus der API-Antwort
answer = response.choices[0].text.strip()


print(answer)"""



# Funktion zum Generieren einer Chatantwort
def generate_chat_response(prompt):
    print("test")
    response = openai.Completion.create(
        engine='text-davinci-003',  # Wählen Sie das gewünschte Modell, z.B. text-davinci-003
        prompt=prompt,
        max_tokens=100,  # Maximale Anzahl der generierten Tokens
        temperature=0.7,  # Steuerung der Kreativität des Modells (0.2 für konservativeres Verhalten, 1.0 für experimentelleres Verhalten)
        n=1,  # Anzahl der generierten Antwortvorschläge
        stop=None,  # Optionale Liste von Stop-Wörtern, um die Antwort zu begrenzen
        timeout=10  # Maximale Dauer für die Generierung der Antwort in Sekunden
    )

    if response.choices:
        return response.choices[0].text.strip()
    else:
        return None

# Beispielanwendung
prompt = text1
response = generate_chat_response(prompt)
print("ChatGPT: " + response)










"""
import pyttsx3

# Initialisiere den Text-to-Speech-Engine
engine = pyttsx3.init()

# Ändere die Stimme zur "Microsoft Hedda"-Stimme
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id) #einzige deutsche Stimme

# Konvertiere Text in Sprache
engine.say("Guten Tag, mein Name ist Heyex. Ich werde sie heute betreuen und sie werden nichts bereuen.")

engine.say("Bitte warten sie noch einen Moment")


#engine.say("Hallo, mein Name ist Heymex. Darf ich ihre Sauerstoffsättigung messen?")

#engine.say("Tom, lauf nicht weg! Du weist, dass es kein Entkommen vor mir, den einzig wahren Heymex, gibt!")

#engine.say("Lasst mich Arzt, ich bin durch!")


names = ["Timo", "Tom", "David"]
for name in names:
    engine.say("Patient " + name +": Sie sind an der Reihe für eine routinemäßige Kontrolluntersuchung.")
               
engine.say("Bitte reichen sie mir ihre Hand und lassen sie sich von mir ins Hinterzimmer führen.")

engine.say("Wir führen nun einige Tests durch!")

engine.say("Zunächst messen wir einmal kurz die Sauerstoffsättigung in ihrem Blut!")
engine.say("Bitte legen sie nun diesen Kompressionsverband an, damit ich ihren Blutdruck messen kann. Sollten sie etwas ungewöhnliches fühlen, sagen sie bitte bescheid!")

engine.say("Nun werden wir noch ihren Blutzucker messen, wenn sie einverstanden sind. Strecken sie hierfür ihre Hand aus!")

engine.say("Ich weis nicht mehr, was ich tun soll, dafür werd ich nicht bezahlt! süüüüü")


#engine.say("R.I.P. an die Singularität der Füchse und Kartoffeln!")
#engine.say("Nix getan, nur eine Bier getrunke")

# Warte, bis die Konvertierung abgeschlossen ist
engine.runAndWait()

# Gib die verfügbaren Stimmen aus
voices = engine.getProperty('voices')
for voice in voices:
    print(f"Stimme: {voice.name}")
    print(f" - ID: {voice.id}")
    print(f" - Geschlecht: {voice.gender}")
    print(f" - Alter: {voice.age}")
    """