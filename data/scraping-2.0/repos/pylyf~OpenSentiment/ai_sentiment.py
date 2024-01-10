# File for handling OpenAI API requests and prompts
import openai
import configparser


# Load the OpenAI API secret key from the config.ini file
config = configparser.ConfigParser()
config.read("config.ini")
secret_key = config.get("OpenAI", "SecretKey")
if secret_key == None:
  print("Please set your API key in the config.ini file. Exiting...")
  exit()
# ----------- #


def openai_analyse_news(news_titles):
  try:
    openai.api_key = secret_key # Load OpenAI API key

    # Prepare the OpenAI Prompt
    openai_prompt = str(news_titles) + " Write if sentiment of these articles is positive (1) neutral (0) or negative (-1), put data in a python list without a variable"
    response = openai.Completion.create(model="text-davinci-003", prompt=openai_prompt, temperature=0, max_tokens=1000)
    # temperature = 0 <- tells chatgpt to keep the answers always as similar as possible, no improvisation
    # max_tokens <- sets the maximum limit of the tokens the prompt can take to prevent speding too much money
    
    response_text = response["choices"][0]["text"]
    rated_news = eval(response_text)

    return rated_news
  except Exception as e:
    return "error: " + str(e)
    

if __name__ == "__main__":
    # This part of code gets executed only if the file gets run as a standalone script, not when it is imported as a library
    print("[OpenSentiment Library]")
    print("-------------------------")
    print("[!] Please run only as an imported library in the main.py file [!]")
    print("Exiting...")
    exit()