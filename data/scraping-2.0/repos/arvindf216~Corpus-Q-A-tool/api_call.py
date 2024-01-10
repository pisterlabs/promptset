from openai import OpenAI
import sys

client = OpenAI(api_key=sys.argv[1])

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python3 api_call.py API_KEY num_paragraphs query.txt")
        sys.exit(1)

    #api_key = sys.argv[1]
    #raise Exception("The 'openai.api_key' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(api_key=api_key)'")  # Set the OpenAI API key

    num_paragraphs = int(sys.argv[2])
    # print(num_paragraphs)

    paragraphs = []

    for i in range(num_paragraphs):
        filename = 'paragraph_' + str(i) + '.txt'
        # print(filename)
        with open(filename, 'r') as f:
            s = f.read()
            paragraphs.append(s)
            paragraphs.append('\n')
            # print(s)
            # print("############")

    query_file = sys.argv[3]
    with open(query_file, 'r') as f:
        query = f.read()
        paragraphs.append(query)
        paragraphs.append('\n')
        # print(query)

    paragraphs = '\n'.join(paragraphs)

    # print(paragraphs)

    query = {"role": "user", "content": paragraphs}

    chat = client.chat.completions.create(model="gpt-3.5-turbo",messages=[query])

    reply = chat.choices[0].message.content
    print("CHATGPT response received\n")
    with open(sys.argv[4], "a") as f:
        f.write(reply)
    # print(reply)
