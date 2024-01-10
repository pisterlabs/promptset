import openai
import sys


if __name__ == '__main__':
    
    # python3 <filename> API_KEY num_paragraphs query.txt
    if len(sys.argv) < 4:
        print("Usage: python3 api_call.py API_KEY num_paragraphs query.txt")
        sys.exit(1)
    
    # Read the API key from the command line
    openai.api_key = sys.argv[1]
    num_paragraphs = int(sys.argv[2])
    #print(num_paragraphs)

    paragraphs = []
    
    # Read the paragraphs from the files
    for i in range(num_paragraphs):
        filename = 'paragraph_' + str(i) + '.txt'
        #print(filename)
        with open(filename, 'r') as f:
            paragraphs.append(f.read())
            paragraphs.append('\n')
    
    # add query
    query_file = sys.argv[3]
    with open(query_file, 'r') as f:
        query = f.read()
        paragraphs.append(query)
        paragraphs.append('\n')

    # convert paragraphs to a single string
    paragraphs = '\n'.join(paragraphs)
    choice=input("\nEnter 1 to see paragraphs else press anything: ")
    if(choice=="1"):
        print(paragraphs)
    print("=================================\nChatGPT's Answer:", end=" ")

    query = {
        "role": "user", "content": paragraphs
    }

    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[query]
    )

    reply = chat.choices[0].message.content
    print(reply)
    print("=================================")
    print("Enter new query: ")