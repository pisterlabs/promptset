#import SpeechToText
#import pyttsx3
#import openai
#import random
from xarm_controller import XArmCtrler, pprint

#openai.api_key_path = r"D:\OneDrive - The Hong Kong Polytechnic University\Interests\cloud_Codes\Python\openai_api_key"

"""
def get_command(user_input: str):
    #Extract command from user input with the help of ChatGPT. Return the command and its arguments.

    def ask_chatgpt(prompt: str) -> str:
        #Fetch reply for `prompt` from ChatGPT.

        pprint("Connecting to ChatGPT...")

        with open("chatGPT-pretrain.txt", "r") as f:
            pretrain_prompt = f.read()

        apologize_list = [
            "Thank you for your patience. I just need a few moments to resolve the issue.",
            "I appreciate your understanding. I need to check on something quickly.",
            "Please bear with me for just a moment. I'm working on a solution.",
            "I apologize for the wait. I'm still working to find a resolution for you.",
            "Thanks for waiting. I just need a little more time to get this sorted.",
            "I'm sorry for the delay. I'm doing everything I can to help you as soon as possible.",
            "I appreciate your time. Please hold on for a few more minutes while I investigate.",
            "Thank you for your patience. I'm checking on something for you and will be right back.",
            "I'm sorry for the inconvenience. Let me take a moment to look into this further.",
            "Thanks for holding. I'm working to find a solution and will be with you shortly.",
        ]
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": pretrain_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                break
            except Exception as e:
                prompt = random.choice(apologize_list)
                speak(prompt)

        text_response = response["choices"][0]["message"]["content"]

        return text_response

    reply = ask_chatgpt(user_input)
    last_line = reply.lower().split("\n")[-1]  # last line is the command sentence
    cmd = last_line.split(" ")[0]  # first word of last line is the command
    pprint("ChatGPT: {}\nInterpreted command: {}".format(reply, cmd))

    if cmd == XArmCtrler.WRITE:  # if last_line is "write [keywords]", return keywords
        kw_start_idx = reply.find("[") + 1
        kw_end_idx = reply.find("]")
        keywords = reply.lower()[kw_start_idx:kw_end_idx]
        return XArmCtrler.WRITE, keywords
    elif cmd == XArmCtrler.ERASE:  # if last_line is "erase", return "erase"
        return XArmCtrler.ERASE, None
    elif cmd == XArmCtrler.PAINT:  # if last_line is "paint", return "paint
        return XArmCtrler.PAINT, None
    elif cmd == XArmCtrler.QUIT:  # if last_line is "quit", return "quit"
        return XArmCtrler.QUIT, None
    elif cmd == XArmCtrler.RESET:  # if last_line is "reset", return "reset"
        return XArmCtrler.RESET, None
    else:  # normal chat with user
        return XArmCtrler.NORMAL_CHAT, reply
"""
"""
def speak(text, end="\n"):
    engine = pyttsx3.init()
    engine.setProperty("voice", "en")
    engine.say(text)
    pprint("xArm: " + text, end=end)
    engine.runAndWait()
"""
"""
def say_goodbye():
    #Say goodbye to user.

    goodbye_list = [
        "Take care and have a great day!",
        "It was great to see you, have a good one!",
        "Thanks for stopping by, see you soon!",
        "Goodbye, and remember to smile!",
        "Bye for now, and don't forget to come back and say hello again!",
        "Farewell, and have a wonderful day!",
        "It was lovely to chat with you, have a safe journey home!",
        "Take it easy, and see you soon!",
        "Have a good one, and keep spreading kindness wherever you go",
    ]

    speak(random.choice(goodbye_list))
"""

def main():
    """Main function."""
    xArm = XArmCtrler("192.168.1.210")
    #word = "WRITINGHEIGHT Z267.9 ABOVEHEIGHT Z280"
    #xArm.write(word)
    xArm.paint()
    #xArm.erase()
    """
    try:
        speak("Hi, I'm xArm. What can I do for you?")
        # Word spliter function demonstration
        while True:
            user_words = SpeechToText.main()
            pprint("You said: {}".format(user_words))

            # Ask ChatGPT to generate a command
            cmd, cmd_param = get_command(user_words)

            if cmd == XArmCtrler.QUIT:
                pprint("Quiting the system...")
                say_goodbye()
                break
            elif cmd == XArmCtrler.ERASE or XArmCtrler.WHITEBOARD_IS_FULL:
                prompt = (
                    "The whiteboard is full. I will erase it first."
                    if XArmCtrler.WHITEBOARD_IS_FULL
                    else "Start erasing!"
                )
                speak(prompt)
                xArm.erase()
                XArmCtrler.WHITEBOARD_IS_FULL = False
                speak("Finish erasing! What else can I do for you?")
            elif cmd == XArmCtrler.PAINT:
                speak("Cheeze!")
                xArm.paint()
                speak("Finish painting! What else can I do for you?")
            elif cmd == XArmCtrler.WRITE:
                speak("Start writing!")
                xArm.write(cmd_param)
                speak("Finish writing! What else can I do for you?")
            elif cmd == XArmCtrler.RESET:
                speak("Resetting...")
                xArm.reset_arm()
                speak("Reset successfully! What else can I do for you?")
            elif cmd == XArmCtrler.NORMAL_CHAT:
                speak(cmd_param)
            else:
                raise Exception("Unknown command: {}, param: {}".format(cmd, cmd_param))
    except KeyboardInterrupt as e:
        pprint("Quit by KeyboardInterrupt")
    """


if __name__ == "__main__":
    main()
