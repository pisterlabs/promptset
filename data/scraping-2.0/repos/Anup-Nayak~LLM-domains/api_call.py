import openai
import sys


if __name__ == '__main__':
    
    # python3 <filename> API_KEY paragraph_0.txt paragraph_1.txt paragraph_2.txt query.txt
    if len(sys.argv) < 5:
        print("Usage: python3 api_call.py API_KEY filename paragraph_0.txt paragraph_1.txt paragraph_2.txt query.txt")
        sys.exit(1)

    # Read the API key from the command line
    openai.api_key = sys.argv[1]

    # Read the paragraphs from the files
    paragraphs = []

    for i in range(3, len(sys.argv)):
        with open(sys.argv[i], 'r') as f:
            paragraphs.append(f.read())
    with open(sys.argv[2], 'w') as fil:
        fil.write("QUERY:\n\n")
        fil.write(paragraphs[len(paragraphs)-1])
        fil.write('\n')

    # convert paragraphs to a single string
    paragraphs = '\n'.join(paragraphs)

    query = {
        "role": "user", "content": paragraphs
    }

    chat = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[query]
    )

    reply = chat.choices[0].message.content
    with open(sys.argv[2], 'a') as f:
        f.write("\nANSWER:\n\n")
        f.write(reply)
    print(reply)

