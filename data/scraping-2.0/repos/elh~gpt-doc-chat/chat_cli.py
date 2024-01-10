import sys
import os
import datetime
import argparse
from dotenv import load_dotenv
import openai

# text-davinci-003 is 10x the cost for chat workflows but has support for fine tuning
MODEL = "text-davinci-003"
LOGS_DIRECTORY = "logs/chat"

modes = {
    "therapy": {
        "prompt": "The following is a conversation with a thoughtful, Rogerian therapist AI, 'CR'. 'CR' is positive and supportive but generally does not do much restating or mirroring. 'CR' is focused on understanding the client's 'incongruence'.",
        "bot_name": "CR"
    },
    "finance": {
        "prompt": "This is a chat with personal financial advisor '$$!'. '$$!' is professional and responds with concise, actionable, informative recommendations.",
        "bot_name": "$$!"
    }
}

def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    parser = argparse.ArgumentParser(description='Chat with AI')
    parser.add_argument('--mode', type=str, default="", help='modes: ' + ", ".join(modes.keys()))
    args = parser.parse_args()
    if args.mode not in modes:
        print("ERROR: Invalid mode. Valid modes: " + ", ".join(modes.keys()))
        sys.exit(1)

    mode = modes[args.mode]
    prompt = mode["prompt"] + "\n\n" # prompt will grow and contain the entire convo so far
    user_input_prompt = "YOU: "
    bot_output_prompt = mode["bot_name"] + ": "

    print("📎")
    print(mode["prompt"] + "\n\n")

    if not os.path.exists(LOGS_DIRECTORY + "/" + args.mode):
        os.makedirs(LOGS_DIRECTORY + "/" + args.mode)
    now = datetime.datetime.now()
    log_file_name = LOGS_DIRECTORY + "/" + args.mode + "/" + str(now.strftime("%Y_%m_%d_%H.%M.%S")) + ".txt"

    total_tokens = 0

    # chat loop
    while True:
        user_input = input(user_input_prompt).strip()
        prompt += user_input_prompt + user_input + "\n" + bot_output_prompt

        print(bot_output_prompt, end="")
        resp = openai.Completion.create(model=MODEL, prompt=prompt, max_tokens=256)
        output = resp.get("choices", [{}])[0].get("text", "").lstrip('\n').lstrip(' ')
        if output == "":
            print("ERROR: No response from OpenAI 🤖\n" + resp)
            sys.exit(1)
        total_tokens += resp.get("usage", {}).get("total_tokens", 0)

        prompt += output + "\n\n"
        print(output + "\n")

        open(log_file_name, 'w').write(prompt + "----------\n" + str(total_tokens) + " tokens. model: " + MODEL) # full reset of file

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\n\nGoodbye 👋')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
