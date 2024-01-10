import openai
import re

def useLocalLLM(host,port):
    openai.api_key = "..."
    openai.api_base = "http://" + host + ":" + port + "/v1"
    openai.api_version = "2023-05-15"

def promptOpenAI(input):
    summary = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-16k',
        # model='llama-2-7b-chat.Q4_0.gguf',
        messages=[{"role":"user", "content": input}]
    )
    return summary.choices[0].message.content + " "


def loadOpenAIKey(keyfile):
    try:
        with open(keyfile, 'r') as f:
            api_key = f.readline().strip()
        return api_key

    except FileNotFoundError:
        print("Key file not found. Please make sure the file exists.")

    except Exception as e:
        print("An error occurred opening the API key file: ", e)

def removeEmojis(text):
    # Define the emoji pattern
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def cleanForTTS(text):
    validChars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!-_$:+-/ ")
    cleanText = ''.join(c for c in text if c in validChars)
    return cleanText