from openai import OpenAI
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import os
from dotenv import load_dotenv
import queue
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
import numpy as np
load_dotenv()

openai_api_key = os.environ['OPENAI_API_KEY']

client = OpenAI()
llm = ChatOpenAI(model="gpt-3.5-turbo-1106",temperature=0,max_tokens=50)

def reply(result_queue):
    while True:
        
        print("Waiting to get item from queue.")
        result = result_queue.get()
        print(f"Retrieved item from queue: {result}")

        completion = client.chat.completions.create(
            model='gpt-3.5-turbo-1106',
            temperature = 0,
            max_tokens=150,
            messages=[
                {"role": "user", "content": f"{result}"}
            ]
        )

        answer = completion.choices[0].message.content
        try:
            answer = completion.choices[0].message.content
            mp3_obj = gTTS(text=answer, lang="en", slow=False)
        
        except Exception as e:
            choices = [
                "I'm sorry, I don't know the answer to that",
                "I'm not sure I understand",
                "I'm not sure I can answer that",
                "Please repeat the question in a different way"
            ]
            mp3_obj = gTTS(text=choices[np.random.randint(0, len(choices))], 
                           lang="en", slow=False)
            print(e)
            


        mp3_obj.save("reply.mp3")
        reply_audio = AudioSegment.from_mp3("reply.mp3")
        play(reply_audio)
        # os.remove("reply.mp3")


