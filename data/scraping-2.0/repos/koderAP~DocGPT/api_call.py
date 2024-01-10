import openai
import sys

if __name__ == '__main__':
    


    if len(sys.argv) < 5:
        print("Usage: python3 api_call.py API_KEY paragraph_0.txt paragraph_1.txt paragraph_2.txt query.txt")
        sys.exit(1)

    # Read the API key from the command line
    openai.api_key = sys.argv[1]

    # Read the paragraphs from the files
    paragraphs = ''
    question = ''

    for i in range(2, len(sys.argv)-1):
        with open(sys.argv[i], 'r') as f:
            paragraphs += (f.read())

    with open(sys.argv[len(sys.argv)-1], 'r') as f:
        question += (f.read())
  

    query = [
        {"role": "system", "content": f"I will provide you with a few paragraphs from the data bank and a Question based on that.\n1 : You are supposed to tell the answer to that question 'from the paragraph only'. 'DONOT ANSWER ANYTHING OUT OF PARAGRAPH, IF NOTHING WITH RESPECT TO QUERY IS AVAILABLE, THEN JUST SAY NOT AVAILAB'Assume your answer would be published as it is in an international newspaper\n2 : Be precise in your response.\n3 : Always mention important points such as dates, names, places, etc. if present. \nTone : It should be formal, concise and understandable to a 5th grade student.\n Give a constructive reply in about 200 words.\n Always give answer in from of points.\n Here is your Question {question}"},
        {"role": "user", "content": paragraphs}
    ]
    

    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages= query,
        temperature = 0
    )

    reply = ""
    print()
    for i in chat.choices:
        reply += i.message.content
        reply += "\n"
    print(reply)




