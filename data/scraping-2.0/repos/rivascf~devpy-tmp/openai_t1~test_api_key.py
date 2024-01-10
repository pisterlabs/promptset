import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def main():
    try:
        if len(openai.api_key) == 51:
            response = openai.Completion.create(model="text-davinci-003", prompt="Say this is a test", temperature=0, max_tokens=7)
        else:
            print(f"API KEY ({len(openai.api_key)})")    
    except:
        print("error")



if __name__ == "__main__":
    main()