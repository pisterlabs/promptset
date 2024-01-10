# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 13:39:19 2023

@author: marca
"""

from openai_pinecone_tools import *
from literature_agent import *
from doc_agent import *
from ingester import *
from google_search import *
from math_agent import *
from table_agent import *
from headmaster_agent import *
from science_agent import *
from tqdm.auto import tqdm
import threading
import time
import sys


class Spinner:
    def __init__(self, message="Thinking..."):
        self._message = message
        self._running = False
        self._spinner_thread = None

    def start(self):
        self._running = True
        self._spinner_thread = threading.Thread(target=self._spin)
        self._spinner_thread.start()

    def stop(self):
        self._running = False
        self._spinner_thread.join()

    def _spin(self):
        spinner_chars = "|/-\\"
        index = 0

        while self._running:
            sys.stdout.write(
                f"\r{self._message} {spinner_chars[index % len(spinner_chars)]}"
            )
            sys.stdout.flush()
            time.sleep(0.1)
            index += 1

        # Clear the spinner line
        sys.stdout.write("\r" + " " * (len(self._message) + 2))
        sys.stdout.flush()
        

# Function to play ElevenLabs prompts
def play_prompt(prompt):
    text_to_speech(prompt)


def main_reg():
    model = FAST_CHAT_MODEL

    spinner = Spinner()
    
    google_assist = False

    print("\n\nWelcome to Study Botty!")

    def add_documents():
        folder_path = input("\nEnter the folder path: ")
        doc_chunks = ingest_folder(folder_path)
        doc_df = create_embeddings_dataframe(doc_chunks)
        store_embeddings_in_pinecone(dataframe=doc_df)

    add_docs = input("\n\nWould you like to add a folder of documents? (y/n): ").lower()
    
    wants_google = input("\nDo you want to use Google Assist? (y/n): ")

    if add_docs == "y":
        add_documents()
        
    if wants_google == "y":
        google_assist = True
    else:
        google_assist = False

    print("\n\nOk, I'm ready for your questions!\n")

    while True:
        query = input(
            "\nEnter your question or enter a command (type 'help' for command list): "
        )

        if query.lower() == "exit":
            break

        elif query.lower() == "smart tutor":
            model = SMART_CHAT_MODEL
            continue

        elif query.lower() == "simple tutor":
            model = FAST_CHAT_MODEL
            continue

        elif query.lower() == "add docs":
            add_documents()
            continue

        elif query.lower() == "help":
            print(
                """
                  Commands:
                      
                    add docs: takes you back to allow another document folder to be stored
                    
                    smart tutor: switch to GPT-4 for question-answering
                                        (warning: more expensive, use only for nuanced questions)
                    
                    simple tutor: switch back to GPT-3.5-turbo for question-answering
                                        (good for most questions)
                                        
                    exit: close the program
                                          
                  """
            )
            continue


        # Start spinner with the default description
        spinner.start()

        # Retrieve the context from Pinecone
        context = fetch_context_from_pinecone(query)

        faculty = headmaster_agent(query)

        if faculty == "DocAgent":
            answer = doc_agent(query, context, model=model)
        elif faculty == "TableAgent":
            answer = table_agent(query, context, model=model)
        elif faculty == "MathAgent":
            answer = math_agent(query, context, model=model)
        elif faculty == "LiteratureAgent":
            answer = literature_agent(query, context, model=model)
        elif faculty == "ScienceAgent":
            answer = science_agent(query, context, model=model)
            
        if google_assist:

            # Check if ChatGPT answered the query with the given context
            did_answer = answer_decision_agent(query, context, answer)
    
            if not did_answer:
                print("\nCould not find answer from context, searching Google...\n")
    
                # If ChatGPT cannot answer with the given context, use the google_search_agent
                context = google_search_agent(query)
    
                faculty = headmaster_agent(query)
    
                if faculty == "DocAgent":
                    answer = doc_agent(query, context, model=model)
                elif faculty == "TableAgent":
                    answer = table_agent(query, context, model=model)
                elif faculty == "MathAgent":
                    answer = math_agent(query, context, model=model)
                elif faculty == "LiteratureAgent":
                    answer = literature_agent(query, context, model=model)
                elif faculty == "ScienceAgent":
                    answer = science_agent(query, context, model=model)

        # Stop the spinner before printing the answer
        spinner.stop()
        print(f"\nAnswer: {answer}")
        
        
        
def main_assisted():
    model = FAST_CHAT_MODEL

    spinner = Spinner()

    google_assist = False


    play_prompt("\n\nWelcome to Study Botty!")

    def add_documents():
        play_prompt("\nEnter the folder path: ")
        folder_path = listen()
        doc_chunks = ingest_folder(folder_path)
        doc_df = create_embeddings_dataframe(doc_chunks)
        store_embeddings_in_pinecone(dataframe=doc_df)

    play_prompt("\n\nWould you like to add a folder of documents?")
    add_docs = listen().lower()

    play_prompt("\nDo you want to use Google Assist?")
    wants_google = listen().lower()

    if 'yes' in add_docs:
        add_documents()

    if 'yes' in wants_google:
        google_assist = True
    else:
        google_assist = False

    play_prompt("\n\nOk, I'm ready for your questions!\n")

    while True:
        play_prompt("\nEnter your question or enter a command. Say 'help' for command list.")
        query = listen()

        if 'exit' in query.lower():
            play_prompt("Goodbye!")
            break

        elif "smart tutor" in query.lower():
            model = SMART_CHAT_MODEL
            play_prompt("Switched to GPT-4.")
            continue

        elif "simple tutor" in query.lower():
            model = FAST_CHAT_MODEL
            play_prompt("Switched back to GPT-3.5.")
            continue

        elif "add docs" in query.lower():
            add_documents()
            continue

        elif 'help' in query.lower():
            play_prompt(
                """
                Commands:
                
                add docs: takes you back to allow another document folder to be stored
                
                smart tutor: switch to GPT-4 for question-answering
                                (warning: more expensive, use only for nuanced questions)
                
                simple tutor: switch back to GPT-3.5-turbo for question-answering
                                (good for most questions)
                                
                exit: close the program
                
                """
            )
            continue

        # Start spinner with the default description
        spinner.start()
        play_prompt("Thinking")
        
        answer = None

        # Retrieve the context from Pinecone
        context = fetch_context_from_pinecone(query)

        faculty = headmaster_agent(query)

        if faculty == "DocAgent":
            answer = doc_agent(query, context, model=model)
        elif faculty == "TableAgent":
            answer = table_agent(query, context, model=model)
        elif faculty == "MathAgent":
            answer = math_agent(query, context, model=model)
        elif faculty == "LiteratureAgent":
            answer = literature_agent(query, context, model=model)
        elif faculty == "ScienceAgent":
            answer = science_agent(query, context, model=model)

        if google_assist:

            # Check if ChatGPT answered the query with the given context
            did_answer = answer_decision_agent(query, context, answer)

            if not did_answer:
                play_prompt("\nCould not find answer from context, searching Google...\n")

                # If ChatGPT cannot answer with the given context, use the google_search_agent
                context = google_search_agent(query)

                faculty = headmaster_agent(query)

                if faculty == "DocAgent":
                    answer = doc_agent(query, context, model=model)
                elif faculty == "TableAgent":
                    answer = table_agent(query, context, model=model)
                elif faculty == "MathAgent":
                    answer = math_agent(query, context, model=model)
                elif faculty == "LiteratureAgent":
                    answer = literature_agent(query, context, model=model)
                elif faculty == "ScienceAgent":
                    answer = science_agent(query, context, model=model)
                   
        # Stop the spinner before printing the answer
        spinner.stop()
        answer_text = f"\nAnswer: {answer}"
        print(answer_text)
        play_prompt(answer_text)
        
def main():

    print("Would you like to use Enhanced Accessibility?")
    
    
    wants_assist = input("Enter your response or say it aloud: ")
    
    # If the user didn't type anything, try speech input
    if wants_assist.strip() == "":
        play_prompt("Would you like to use Enhanced Accessibility?")
        wants_assist = listen()
    
    if any(word in wants_assist.lower() for word in ["yes", "yep", "sure", "ok"]):
        main_assisted()
        
    else:
        main_reg()



if __name__ == "__main__":
    main()
