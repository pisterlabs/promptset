#this is a script that uses gpt to condense a chat log. The chatlog is obtained from telegram
import openai
import datetime
TOKEN = open("telegram_token.txt", "r").read()

openai.api_key = open("openai_key.txt", "r").read()
#current time
currTime = datetime.datetime.now()

def condense_chat(chat_id="1085483721"):
    global currTime
    condensed_chatlog = ""
    if(datetime.datetime.now() - currTime).seconds > 1000:
        currTime = datetime.datetime.now()
        chatlog = open(chat_id+"log.txt", "r").read()
        #print the chatlog
        #create a chunk for each line
        chunks = chatlog.split("\n")
        #combine every 30 elements into one string
        chunks = [" ".join(chunks[i:i+30]) for i in range(0, len(chunks), 30)]
        #remove empty strings
        chunks = [chunk for chunk in chunks if chunk != ""]

        #condense each chunk
        condensed_chunks = []
        for chunk in chunks:
            condensed_chunk = openai.Completion.create(
                engine="text-davinci-003",
                prompt="Condense the following chat to minimize word count. What all happened in the chat, by whom, and when? Represent older data less and newer data more: \n"+chunk+"\n Condensed: ",
                max_tokens=1000,
                n=1,
                stop=None,
                temperature=0,
            ).get("choices")[0].text
            condensed_chunks.append(condensed_chunk)
        #join the condensed chunks
        condensed_chatlog = "".join(condensed_chunks)
        #save to file
        with open(chat_id+"condensed_chatlog.txt", "w") as f:
            f.write(condensed_chatlog)
    with open(chat_id+"condensed_chatlog.txt", "r") as f:
        condensed_chatlog = f.read()
    return condensed_chatlog
condensed_chatlog = condense_chat()
print(condensed_chatlog)
