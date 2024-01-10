import openai
import os
import sys

openai.api_key = os.getenv("OPENAI_API_KEY")
is_debug = False
is_simulation = False

def read_in_pieces(file_object, piece_size=13500):
  words = []
  for line in file_object:
    words += line.split()
    while len(words) >= piece_size:
      yield ' '.join(words[:piece_size])
      words = words[piece_size:]
  yield ' '.join(words)

def debug(msg):
  if is_debug:
    print(msg)

def usage():
  if is_debug:
    print(f"Usage: python3 {sys.argv[0]} SOURCE_FILE")
  
def main():
  if len(sys.argv) > 1:
    source_file = sys.argv[1]
  else:
    usage()
    sys.exit(1)

  with open(source_file) as f:
    for i, piece in enumerate(read_in_pieces(f)):
      debug(f"\n\npiece {i}:\n\n{piece}\n")
      prompt=f"Code snippet:\n{piece}\n\n\"\"\"Here's what the above code is doing, explained in bullet points concisely:\n+ "
      messages = [
        {"role": "system",
         "content": "your are a intelligent software engineer who is good at summarize code in concise way."},]
      messages.append({"role": "user", "content": prompt},)
      #debug(f"\n\nprompt {i}:\n\n{prompt}\n")

      if is_simulation:
        continue

      response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        temperature=0,
        max_tokens=500,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\"\"\""]
      )
      print(response.choices[0].message.content.strip())
      
if __name__ == "__main__":
  main()
