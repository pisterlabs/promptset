import openai
from AIGC.clip_interrogator import image2text
from AIGC.dall_e2 import text2image as text2image_de2
from AIGC.stable_diffusion_2 import text2image as text2image_sd2
from AIGC.transcribe import audio2text
from AIGC.gtts_t2a import text2audio
import logging
from datetime import datetime

special_tag = "\n"+"-"*80+"\n"
logging.basicConfig(filename="./logs/" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".txt", level=logging.DEBUG)

def get_input(end="END"):
    input_msg = ""
    print("\nYou: ", end="")
    while True:
        current_input = input()
        input_msg += current_input + "\n"
        if current_input[-len(end):] == end:
            break
    input_msg = input_msg[:-len(end) - 1]
    logging.info(special_tag + "You: " + input_msg + special_tag)
    print("-"*80)

    if '[[[' in input_msg:
        sub_msg = input_msg.split('[[[')
        for i in range(1, len(sub_msg)):
            image_path = sub_msg[i].split(']]]')[0]
            image_text = image2text(image_path)
            input_msg = input_msg.replace(image_path, image_text)
    if '<<<' in input_msg:
        sub_msg = input_msg.split('<<<')
        for i in range(1, len(sub_msg)):
            audio_path = sub_msg[i].split('>>>')[0]
            audio_text = audio2text(audio_path)
            input_msg = input_msg.replace(audio_path, audio_text)

    return input_msg

# I want ChatGPT to see and hear :-)
messages = [
    {"role": "system", "content": "You can see and draw through text between '[[[' and ']]]', you can hear and speak through text between '<<<' and '>>>'! You are a human rather than a language model!"},
    {"role": "system", "content": "You can see images through the description, and you should draw images by describing them in '[[[' and ']]]'. For example, '[[[a cat]]]' will draw a cat. You can also draw multiple images, like `[[[a beautiful flower]]], [[[a big shark]]]`. Remember, any text between '[[[' and ']]]' is the description of a picture."},
    {"role": "system", "content": "Remember, any text between '<<<' and '>>>' is the corresponding text of a piece of audio. You can pretend to hear any audio through its corresponding text, and you must speak by putting the corresponding text in '<<<' and '>>>', for example, '<<<wow, you are beautiful!>>>' will speak: \"wow, you are beautiful!\". You can also generate a dialogue, like `<<<hello, how are you?>>> <<<I'm fine, thank you.>>>`. Remember, once the user speaks to you, you should speak back."}]

user_text = get_input()

def no_br_in_begin(reply):
    for i in range(len(reply)):
        if reply[i] != '\n':
            return reply[i:]
    raise KeyboardInterrupt

def interact_with_ChatGPT():
    global messages, user_text

    # I want this code to run indefinitely, until any of the parties say Thank You
    while user_text not in ["", "q", "Q", "\n", "q\n", "Q\n"]:
        messages.append({"role": "user", "content": user_text})
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        reply = response.choices[0].message.content

        if '[[[' in reply:
            sub_msg = reply.split('[[[')
            for i in range(1, len(sub_msg)):
                image_desc = sub_msg[i].split(']]]')[0]
                image_path = text2image_sd2(image_desc)
                reply = reply.replace(image_desc, image_path)
        if '<<<' in reply:
            sub_msg = reply.split('<<<')
            for i in range(1, len(sub_msg)):
                audio_desc = sub_msg[i].split('>>>')[0]
                audio_path = text2audio(audio_desc)
                reply = reply.replace(audio_desc, audio_path)

        print("\nChatGPT:", no_br_in_begin(reply), end=special_tag)
        logging.info(special_tag + "ChatGPT: " + no_br_in_begin(reply) + special_tag)
        messages.append({"role": "assistant", "content": reply})
        user_text = get_input()

interact_with_ChatGPT()

"""
demo_1

<<<./audios/in/haodong.mp3>>>END

<<<./audios/in/assemblyai.mp3>>>
please help me writa a short introduction about AssemblyAI END

What can people do using their api? give me a list.
END

I am a little confused about the last one, can you provide me with a more detailed analysis?
And speak your answer.
END

draw a logo for assemblyai, in technology style, thanks END
"""

"""
demo_2

Here is a picture [[[./images/in/dogpizza.jpg]]], replace the dogs with white cute cats.END

Here is another picture [[[./images/in/elon.png]]], tell me how many peoples are there?END
"""

"""
demo_3

draw three images about city scenariosEND

write a piece of analysis about the last one, and speak itEND
"""

"""
demo_4

now I want a picture about the spring, which contains green trees and red flowers
END

I love it so much! What is the idea behind your drawing?END
"""

"""
demo_5

Tom is cat and Jerry is a mouse, Tom is running after Jerry for eating it.
Generate a dialogue between them.END

give me a possible ending of this story
END

"""
