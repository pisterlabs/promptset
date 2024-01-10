import pyttsx3
import speech_recognition as sr
import openai
from api import key


class SpeechRecognition:
    def __init__(self, properties):
        # Set property values
        self.energy_threshold = properties[0]
        self.sample_rate = properties[1]
        self.adjust_for_ambient_noise = properties[2]
        self.non_speaking_duration = properties[3]
        self.timeout = properties[4]
        self.phrase_time_limit = properties[5]
        # Prep Modules
        self._prep_recognizer()
        self._prep_voice()
        self._prep_ai()
        # Begin listening for input
        self._listen()

    def _prep_recognizer(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = self.energy_threshold
        self.recognizer.non_speaking_duration = self.non_speaking_duration

    def _prep_voice(self):
        self.tts = pyttsx3.init()
        voices = self.tts.getProperty('voices')
        self.tts.setProperty('voice', voices[1].id)

    def _prep_ai(self):
        openai.api_key = key
        self.model = "text-davinci-002"  # Ada
        self.max_tokens = 60
        self.temperature = 0.5

    def _listen(self):
        while True:
            with sr.Microphone(sample_rate=self.sample_rate) as source:
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=self.timeout,
                                               phrase_time_limit=self.phrase_time_limit) # noqa

            try:
                text = self.recognizer.recognize_google(audio)
                self._process_text(text)

            except sr.UnknownValueError:
                self._chat("Could not understand audio.")

            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))

    def _get_ai_response(self, prompt: str):
        completions = openai.Completion.create(
            engine=self.model,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,)

        message = completions.choices[0].text.strip()
        return message

    def _process_text(self, text):
        text.lower()
        print("You said:", text)

        # Conditions
        if "please" in text.lower():
            response = self._get_ai_response(text)
            self._chat(response)
        elif "cat" in text.lower():
            self._chat("MEOW")
        elif "dog" in text.lower():
            self._chat("BARK")
        else:
            self._chat("I don't know what you're talking about.")

    def _chat(self, message: str):
        self.tts.say(message)
        print(message)
        self.tts.runAndWait()

# TODO - Need to set it up to retain converastion history.
