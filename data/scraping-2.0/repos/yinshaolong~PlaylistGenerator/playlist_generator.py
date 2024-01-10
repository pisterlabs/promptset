import openai
import os
import dotenv
import argparse
import json

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

model = {3:"gpt-3.5-turbo", 4:"gpt-4-1106-preview"}

example_json = """[
  {"song": "Eye of the Tiger", "artist": "Survivor"},
  {"song": "Stronger", "artist": "Kanye West"},
  {"song": "Don't Stop Believin'", "artist": "Journey"},
  {"song": "Harder, Better, Faster, Stronger", "artist": "Daft Punk"},
  {"song": "Can't Hold Us", "artist": "Macklemore & Ryan Lewis"}
]"""

def get_user_input(prompt:str)->str:
    valid_input = False
    while not valid_input:
        user_input = input(f"No {prompt} given. Please enter a {prompt}: ")
        valid_input = True if user_input else False
    return user_input

def logout_user(args):
    if os.path.exists(".cache") and args.logout:
        os.remove(".cache")
        print("User logged out.")
    elif args.logout:
        pass
        print("User is not logged in.")
    return args.logout

def parse_args():
    parser = argparse.ArgumentParser(description="Generate a playlist based on a prompt")
    parser.add_argument("-p", type=str, help="Prompt to generate playlist from")
    parser.add_argument("-m", default = 4, type=str, help="Model to use for generation")
    parser.add_argument("-l", type=int, help="Length of playlist to generate")
    parser.add_argument("-pop", default = "private", type=str, help="Determines if playlist is public or private")
    parser.add_argument("-t", default = "spotify", type=str, help="Determines playlist type (spotify or youtube)")
    parser.add_argument( "--logout", help="Logs user out by deleting the cache.", action='store_true') # store_true is false without flag
    return parser.parse_args()

def set_prompt_and_length(count, user_prompt):
    messages = get_prompt()
    count = count if count else get_user_input("length of playlist")
    playlist_gen_prompt = f"Generate a playlist of {count} songs based on this prompt:"
    user_prompt = user_prompt if user_prompt else get_user_input("prompt")
    #passing user_prompt down to get_playlist so that we can use as title in app.py
    return messages, playlist_gen_prompt, user_prompt

def get_messages_and_model(length = None, prompt = None)->list[str]:
    args = parse_args()
    gpt_model = model[int(args.m)]
    length = args.l if not length else length
    prompt = args.p if not prompt else prompt
    messages, playlist_gen_prompt, user_prompt = set_prompt_and_length(length, prompt)
    messages[1]["content"] = f"{messages[1]['content']}{'motivation to do homework last minute'}"
    messages[2]["content"] = f"{example_json}"
    messages[3]["content"] = f"{playlist_gen_prompt}{user_prompt}"
    # print("messages: ",messages)
    return messages, gpt_model, user_prompt

def get_reply(messages:str, gpt_model:str)->str:
    for data in client.chat.completions.create(
    model=gpt_model,
     messages=messages,
    # max_tokens = 200,
    stream=True,
    ):          
         #streams have a different format - have delta.content instead of message.content
        content = data.choices[0].delta.content
        if content is not None:
            yield content

def get_prompt(prompt = "prompt.json")->list[dict]:
    with open(prompt, "r") as f:
        return json.load(f)

def get_playlist(length = None, prompt = None)->list[dict]:
    '''
        get json of the prompt.
        in format of: {"song": <song name>, "artist": <artist name>}.
    '''
    playlist_tokens = []
    args = parse_args()
    messages, gpt_model, user_prompt = get_messages_and_model(length, prompt)
    for data in get_reply(messages, gpt_model):
        playlist_tokens.append(data)
        print(data, end="", flush=True)
    playlist = "".join(playlist_tokens)
    playlist = json.loads(playlist)
    return playlist, user_prompt

def main():
    playlist = get_playlist()
    # print("\n\n\nplaylist: ",playlist, "type: ",type(playlist))
if __name__ == "__main__":
    main()