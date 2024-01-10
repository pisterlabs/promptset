import os
import openai
import argparse
import httpx
openai.api_key = os.getenv("OPENAI_API_KEY")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
      "--file",
      help="File to be scanned...",
      required=False,
    )
    parser.add_argument(
      "--url",
      help="Url to be scanned...",
      required=False,
    )

    args = parser.parse_args()
    file = args.file
    url = args.url
    code = []
    if file:
      with open(file) as f:
        for line in f:
          code.append(line.rstrip())
    if url:

      with httpx.Client() as client:
        try:
          headers = {'User-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'}
          r = client.get(url, headers=headers)
          code = r.text
        except:
          pass

    response = openai.Completion.create(
      model="text-davinci-003",
      prompt="I am a highly intelligent code auditing machine. Q: Where is the following code vulnerable: " + str(code) + "\nA:", 
      temperature=0,
      max_tokens=100,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=["\n"]
    )
    answer = response['choices'][0]['text']
    print(answer)

if __name__ == "__main__":
    main()
