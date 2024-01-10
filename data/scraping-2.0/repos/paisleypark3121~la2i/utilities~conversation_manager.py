import os
from dotenv import load_dotenv
from openai import OpenAI

from chromadb_manager import *

def setMessages(messages, rolling):
    num_entries = len(messages)
    num_couples = (num_entries - 1) // 2 

    if num_couples <= rolling:
        return messages  
        
    couples_to_remove = num_couples - rolling

    removed_couples = 0
    index = 1  
    while removed_couples < couples_to_remove:
        if messages[index]["role"] == "user" and messages[index + 1]["role"] == "assistant":
            del messages[index]
            del messages[index] 
            removed_couples += 1
        else:
            index += 1

    return messages

def main_conversation(client,model,temperature,messages,rolling):
    try:
        print("\n***WELCOME***\n")
        while True:
            prompt = input("\nuser: ")
            messages.append(
                {
                    "role":"user",
                    "content": prompt
                }
            )
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=400,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            print(f"assistant: {response.choices[0].message.content}")

            messages.append(
                {
                    "role":"assistant",
                    "content":response.choices[0].message.content
                }
            )
            
            messages=setMessages(messages,rolling)
            # print("***DEBUG MESSAGES START***")
            # print(messages)
            # print("***DEBUG MESSAGES FINISH***")
    except KeyboardInterrupt:
        print("\n***GOODBYE***\n")

def main_conversation_with_context(
    client,
    model,
    temperature,
    system_message,
    messages,
    rolling,
    filename,
    persist_directory):

    vectordb=create_vectordb_from_file(
        filename=filename,
        persist_directory=persist_directory,
        embedding=embedding,
        overwrite=True,
        chunk_size=500,
        chunk_overlap=50)

    retriever=vectordb.as_retriever()
    
    try:
        print("\n***WELCOME***\n")
        while True:
            prompt = input("\nuser: ")
            messages.append(
                {
                    "role":"user",
                    "content": prompt
                }
            )

            docs=retriever.get_relevant_documents(prompt)
            #print(docs[0])
            updated_system_message=system_message.replace("{context}", docs[0].page_content)

            messages[0]["content"]=updated_system_message

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=400,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            print(f"assistant: {response.choices[0].message.content}")

            messages.append(
                {
                    "role":"assistant",
                    "content":response.choices[0].message.content
                }
            )
            
            messages=setMessages(messages,rolling)
            # print("***DEBUG MESSAGES START***")
            # print(messages)
            # print("***DEBUG MESSAGES FINISH***")
    except KeyboardInterrupt:
        print("\n***GOODBYE***\n")

def chromadb_test(filename,persist_directory,embedding,system_message):

    vectordb=create_vectordb_from_file(
        filename=filename,
        persist_directory=persist_directory,
        embedding=embedding,
        overwrite=True,
        chunk_size=500,
        chunk_overlap=50)

    retriever=vectordb.as_retriever()

    question="In which year jokerbirot landed on planet earth?"
    #question="what happened in 2077?"
    #question="Who is william shakespeare?"
    docs=retriever.get_relevant_documents(question)
    
    #[print(d[1]) for d in vectordb.similarity_search_with_score(question, k=5 )]
    #scores=vectordb.similarity_search_with_score("what is quantum computing?")
    #print(scores)
    #print(docs[0])
    #print(retriever.search_type)

    updated_system_message=system_message.replace("{context}", docs[0]["page_content"])
    
system_message="Your role is to be a helpful assistant with a friendly, "\
    "understanding, patient, and user-affirming tone. You should: "\
    "explain topics in short, simple sentences; "\
    "keep explanations to 2 or 3 sentences at most. "\
    "If the user provides affirmative or brief responses, "\
    "take the initiative to continue with relevant information. "\
    "Check for user understanding after each brief explanation "\
    "using varied and friendly-toned questions. "\
    "Use ordered or unordered lists "\
    "(if longer than 2 items, introduce them one by one and "\
    "check for understanding before proceeding), or simple text in replies. "\
    "Provide examples or metaphors if the user doesn't understand. "\
    "Use the following additional [context] below (if present) to retrieve information; "\
    "if you cannot retrieve any information from the [context] use your knowledge. "\
    "[context] {context}"

rolling_limit = 3

messages=[]
messages.append(
    {
      "role": "system",
      "content": system_message
    }
)

load_dotenv()
client=OpenAI()

model=os.environ.get('FINE_TUNED_MODEL')
openai_api_key=os.environ.get('OPENAI_API_KEY')
temperature=0
rolling=3

embedding=OpenAIEmbeddings()

#filename="../files/wilde.txt"
filename="../files/jokerbirot_space_musician_en.txt"
persist_directory = 'chroma'
#chromadb_test(filename,persist_directory,embedding,system_message)

#main_conversation(client,model,temperature,messages,rolling)
main_conversation_with_context(
    client,
    model,
    temperature,
    system_message,
    messages,
    rolling,
    filename,
    persist_directory)