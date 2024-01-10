from openai import OpenAI
import os
import sounddevice as sd
import numpy as np
from pydub import AudioSegment
import io
import stt
os.getenv("OPENAI_API_KEY")
client = OpenAI() # make global so every function has access

# from the python file where stt is happening, pass something here
# write a response function that takes in the return value from the stt function, passes it to gpt 3.5-turbo and gets a completion
# return this and pass it as input to the response which has correct output now. 
def get_gpt4_response(text):
    chat_completion = client.chat.completions.create(
        # create two different roles for the api so we can pass in some knowledge to ground its response
        messages=[
            {
                "role": "system",
                "content": "In addition to being a very knowledgeable person, you are super knowledgeable about The MuNCHER which is a project built by Ian, Kenta, Matt, Mihir, and Zayn. The MuNCHER is a Mars rover replica and is on the path to collect multiple soil samples, autonomously navigate its surroundings, and stream data to a central location. The students have been working on this project for 6 weeks and have built a soil sampler mechanism, drivetrain, suspension system, and plan to take on NASA as the world's best rover. Keep your responses to 40 words or less. Be factual and concise. Don't be afraid to say I don't know."
            },
            {
                "role": "user",
                "content": text
            }
        ],
        model="gpt-3.5-turbo",
        max_tokens=200, 
        temperature=0.5
    )
    print(chat_completion) # this is so I know what to pass into the chat
    
    if chat_completion.choices:
        first_response_content = chat_completion.choices[0].message.content
        print(f' this is the first response content {first_response_content}')
        print(type(first_response_content))
        return first_response_content
    else:
        return ""


def convert_to_good_audio(chat):
    response = client.audio.speech.create(
        model = "tts-1",
        voice = "alloy",
        input = chat
    )

    data = response.content # use this to extract data from the response so I can process it without needing to write it to a file
    audio_segment = AudioSegment.from_file(io.BytesIO(data), format="mp3") # get the binary data from response and format it as an mp3
    samples = np.array(audio_segment.get_array_of_samples())

    sd.play(samples, samplerate=audio_segment.frame_rate)
    sd.wait()