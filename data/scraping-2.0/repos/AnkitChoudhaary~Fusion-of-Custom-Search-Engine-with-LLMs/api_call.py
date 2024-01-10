import openai
import sys
import os

if __name__ == '__main__':
    
    # python3 <filename> API_KEY num_paragraphs query.txt
    if len(sys.argv) < 4:
        print("Usage: python3 api_call.py API_KEY num_paragraphs query.txt")
        sys.exit(1)

    # Read the API key from the command line
    openai.api_key = sys.argv[1]
    num_paragraphs = int(sys.argv[2])

    # Read the paragraphs from the files
    paragraphs = []

    # open output.txt
    output_file = 'output.txt'

    # if file does not exist, create it
    if not os.path.exists(output_file):
        os.system('touch ' + output_file)

    # open file in append mode
    f = open(output_file, 'a')
    f.write("Top " + str(num_paragraphs) + " paragraphs:\n\n")

    c = 0
    for i in range(num_paragraphs):
        filename = 'paragraph_' + str(i) + '.txt'
        with open(filename, 'r') as pf:
            p = pf.read()
            f.write('Paragraph ' + str(i) + ':\n')
            f.write(p)
            f.write('\n')
            f.write('======================================================\n')
            paragraphs.append(p)
            paragraphs.append('\n')
    
    # add query
    query_file = sys.argv[3]
    with open(query_file, 'r') as qf:
        query = qf.read()
        paragraphs.append(query)
        paragraphs.append('\n')

    # convert paragraphs to a single string
    paragraphs = '\n'.join(paragraphs)


    # write paragraphs to output.txt
    # f.write("Sending the following to GPT-3:\n\n")
    # f.write(paragraphs)
    # f.write("==================================================================\n\n")

    query = {
        "role": "user", "content": paragraphs
    }

    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[query]
    )

    reply = chat.choices[0].message.content
    f.write("GPT-3's response:\n\n")
    f.write(reply)
    print(reply)
    f.write("===================================================================\n\n")
    f.close()