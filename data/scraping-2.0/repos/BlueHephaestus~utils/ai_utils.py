"""
Utils to use AI in some way shape or form.
    It's very generic for now.

All of these will assume an API key in the environment variable OPENAI_API_KEY
"""
import nltk # didn't see this functionality for sentence splitting in the openai api
# nltk.download("punkt")
import os
from tqdm import tqdm
from openai import OpenAI
import json

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def tts(text, output_file="output.mp3", model="tts-1-hd", voice="fable", chunk_size=4096):
    """
    Convert text to speech using the OpenAI API, and save it to a file.

    Since OpenAI has a limit on the character limit that can be sent to the API,
    this function will split the text into chunks of full length 4096 and send them individually.

    It splits on sentence boundaries, so that the audio doesn't cut off mid-sentence.

    :param text: The text to convert to speech. Should only be a string.
    :param output_file: The file to save the audio to. Defaults to output.mp3.
    :param model: The model to use. Defaults to tts-1.
    :param voice: The voice to use. Defaults to fable.
    :return: The audio data.
    """

    open(output_file, "w").close()  # clear the file

    def sent_chunk_iter(text):
        chunk = ""
        for sent in tqdm(nltk.sent_tokenize(text), desc="TTS: "):
            if len(chunk + sent) + 1 < chunk_size:
                chunk += sent + " "
            else:
                yield chunk
                chunk = ""
        yield chunk

    # Iterate in chunk_size sentence-delimited chunks until all text processed
    for chunk in sent_chunk_iter(text):

        # Process the chunk and reset it
        response = client.audio.speech.create(model=model, voice=voice, input=chunk)

        # add this chunk to the file
        with open(output_file, "ab") as f:
            for filechunk in response.iter_bytes(chunk_size=1024): # arbitrary chunk size
                f.write(filechunk)

class ChatGPT:
    def __init__(self, id="untitled", system_prompt="You are a helpful assistant.", model="gpt-4"):
        """
        Initialize a new chat session.
        :param id: The id of this chat session. Used to save the session.
        :param system_prompt: The prompt to use for the system.
        :param model: The model to use. Defaults to gpt-4.

        This will start a new chat session with the given parameters, and populate a file with the chat history
            to resume it later if needed.

        No prompts are sent until the first user prompt is given.
        """
        self.id = id
        self.model = model
        self.system_prompt = system_prompt
        self.history_fname = f"prompts/{self.id}.json"
        self.history = [{"role":"system", "content":self.system_prompt}]


    @classmethod
    def from_file(self, id="untitled", model="gpt-4"):
        """
        Alternate constructor to load a chat session from a file and resume.
            Chat session formats are JSON:
            [
                {"role": "system", "content": "System Prompt"}
                {"role": "user", "content": "Who was phone?"}
                {"role": "assistant", "content": "I was phone."}
                {"role": "user", "content": "*gasp*"}
            ]
        """
        # Load the chat history
        self = ChatGPT(id=id, model=model)
        self.history_fname = f"prompts/{self.id}.json"
        self.history = []
        if not os.path.exists(self.history_fname):
            print("No chat history found, starting new chat session.")
            return self

        with open(self.history_fname, "r") as f:
            self.history = json.load(f)
        self.system_prompt = self.history[0]["content"]
        return self

    def cumulative_ask(self,  prompt):
        """
        Send the prompt as the most recent in the string of prompts for this id.
            Prompt will be sent along with all previous prompts in the chat history.
        """
        self.history.append({"role":"user","content":prompt})
        response = client.chat.completions.create(messages=self.history, model=self.model)
        response_content = response.choices[0].message.content
        self.history.append({"role":"assistant","content":response_content})
        return response_content

    def ask(self, prompt):
        """
        Send the prompt to the chat session and return the response, without saving the chat history.
            This means the only thing sent to the model will be the system prompt, and this
            given prompt as a "user" prompt.

        Called like
        chat = ChatGPT(system_prompt="You are Douglas Adams", model="gpt-4")
        # These will be each sent separately
        chat.ask("What is the meaning of life?")
        chat.ask("What is the meaning of the universe?")
        chat.ask("What is the meaning of everything?")

        NONE of these prompts are saved to the chat history.
        """
        #history = [{"role":"system", "content":self.system_prompt}, {"role":"user", "content":prompt}]
        history = [{"role":"system", "content":self.system_prompt}, {"role":"user", "content":prompt}]
        response = client.chat.completions.create(messages=history, model=self.model)
        response_content = response.choices[0].message.content
        return response_content

    def save(self):
        """
        Save the chat history to a file.
        """
        with open(self.history_fname, "w") as f:
            json.dump(self.history, f, indent=2)

if __name__ == "__main__":
    # Test chatgpt asks
    chat = ChatGPT(model="gpt-3.5-turbo", system_prompt="Respond with only one token.")
    for i in range(5):
        print(chat.ask("What is the meaning of life, the universe, and everything?"))

    # Test cumulative asks
    chat = ChatGPT(model="gpt-3.5-turbo", system_prompt="Respond with only one token.")
    print(chat.cumulative_ask("What is the first letter in the phonetic alphabet?"))
    for i in range(5):
        print(chat.cumulative_ask("What is the next letter in the phonetic alphabet?"))
    chat.save()
    chat = ChatGPT.from_file(model="gpt-3.5-turbo")
    for i in range(5):
        print(chat.cumulative_ask("What is the next letter in the phonetic alphabet?"))

