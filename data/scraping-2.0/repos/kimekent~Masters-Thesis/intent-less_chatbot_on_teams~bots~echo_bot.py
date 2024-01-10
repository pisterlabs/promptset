# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
This file contains the code to run the final version of the intent-less chatbot using Microsoft Teams as the front end
user interface.
The question, restructured question, current prompt and response are printed on to the console. If wished
the current prompt print statement can be commented out since this can get pretty long (line 130)

To run this script update the 'path' variable to the root directory of this project and add your OpenAI API key to
'openaiapikey.txt' in the root directory of this project.
"""
# 1. Set up-------------------------------------------------------------------------------------------------------------
# Set path to project directory and define OpenAI API key
from functions.functions import open_file
path = r"C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\Masters-Thesis"

import openai
import os
os.environ["OPENAI_API_KEY"] = open_file(path + r"\openaiapikey.txt")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Import packages
# Token Management and File Operations
import os
from functions.functions import num_tokens_from_string, remove_history, save_file, adjust_similarity_scores

# OpenAI Libraries and functions
import openai
from functions.functions import gpt3_1106_completion
import ast

# Libraries for initializing the retriever and the vector store
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Library and function for sending mails
import win32com.client as win32
from functions.functions import send_email

# Libraries to run bot on MS Teams
from botbuilder.core import ActivityHandler, MessageFactory, TurnContext
from botbuilder.schema import ChannelAccount

# Define directory paths
teams_bot_path = path + r'\intent-less_chatbot_on_teams\bots'
chroma_directory = teams_bot_path + r'\webhelp and websupport_vector_db'
prompt_logs_directory = teams_bot_path + r"\gpt3_logs\prompt"
retriever_prompt_log_directory = teams_bot_path + r"\gpt3_logs\retriever_prompt"

# Initialize Chroma vector database
vectordb_websupport_bot = Chroma(persist_directory=chroma_directory,
                                 embedding_function=OpenAIEmbeddings(model='text-embedding-ada-002'))

# Import list of words to check for re-scoring of retriever similarity scores.
words_to_check = ast.literal_eval(open_file(teams_bot_path + r"/words_to_check_for_adjusted_similarity_score.txt"))

# Remove history from prompt, of previous sessions
question_prompt = remove_history(open_file(teams_bot_path + r"\prompts\question_prompt.txt"),
                                 "HISTORY:(.*?)<<history>>",
                                 "HISTORY: <<history>>")
save_file(question_prompt, teams_bot_path + r"\prompts\question_prompt.txt")

retriever_prompt = remove_history(open_file(teams_bot_path + r"\prompts\retriever_prompt.txt"),
                                  "<<newest message>>(.*?)<<oldest message>>",
                                  "<<newest message>>\n<<history>>\n<<oldest message>>")
save_file(retriever_prompt, teams_bot_path + r"\prompts\retriever_prompt.txt")

# Initialize states and query count for every new session
save_file(str(0), teams_bot_path + r"\query_count.txt")
state = str({'need_help': 'no', 'edit_ticket': 'none'})
save_file(state, teams_bot_path + r"\state.txt")


class EchoBot(ActivityHandler):
    async def on_members_added_activity(
            self, members_added: [ChannelAccount], turn_context: TurnContext
    ):
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity('Hi! Als Web-Support-Chatbot bin ich hier, um dir bei Fragen zum'
                                                 ' Bearbeiten unserer Website zu helfen. Wie kann ich dir heute zur'
                                                 ' Seite stehen?')

    async def on_message_activity(self, turn_context: TurnContext):
        state = ast.literal_eval(open_file(teams_bot_path + r"\state.txt"))
        message = str(turn_context.activity.text)
        print(f"Incoming message: {message}")
        # Set states
        # Check if incoming message is new message that should be sent as ticket
        if state["edit_ticket"] == "yes" and state["need_help"] == "yes":
            new_message_received = "yes"
            print("new_message_received = yes")
        else:
            new_message_received = "no"
            print("new_message_received = no")

        # Check if incoming message is answer to question if user wants to edit question before sending
        if state["edit_ticket"] == "question":
            follow_up_edit_answer = "yes"
            print("followup edit yes")
        else:
            follow_up_edit_answer = "no"
            print("followup edit no")

        # Check if incoming message is answer to question if user wants to create ticket
        if state["need_help"] == "yes":
            follow_up_answer = "yes"
            print("follow_up_answer = yes")

        if state["need_help"] == "no":
            follow_up_answer = "no"
            print("follow_up_answer = no")
            # Load prompts
            # This prompts is used to restructure question for the retriever. For each new session the history is deleted.
            retriever_prompt = open_file(teams_bot_path + r"\prompts\retriever_prompt.txt")

            # This prompt is used to answer to question. For each new session the history is deleted.
            question_prompt = open_file(teams_bot_path + r"\prompts\question_prompt.txt")

            l_retriever_history = []  # This list will contain the previous questions and answers.
            l_qa_history = []  # This list will contain the previous questions and answers for the Q&A prompt.

            print("on_message_activity called")
            message = str(turn_context.activity.text).lower()
            print(f"Incoming message: {message}")

            # Use GPT-3 to generate a response
            query_count_file = teams_bot_path + r'\query_count.txt'
            with open(query_count_file, 'r') as count_file:
                query_count = int(count_file.read())
                print("query count: " + str(query_count))

            # Prompt the user for a question
            query = message
            print("query: " + query)

            if query_count > 1:
                # Reformulate the current question to include context from previous turns for better document retrieval
                # in multi-question sessions.
                current_retriever_prompt = retriever_prompt.replace('<<query>>', query)
                restructured_query = gpt3_1106_completion(prompt=current_retriever_prompt,
                                                          log_directory=retriever_prompt_log_directory)

            else:
                restructured_query = query
            save_file(restructured_query, teams_bot_path + r"\restructured_query.txt")

            print("restructured query: " + restructured_query)
            # Initialize lists per question.
            l_webhelp_articles = []  # This list will contain all retrieved webhelp articles.
            l_webhelp_questions = []  # This list will contain all retrieved websupport questions.

            # Count how many tokens are in the prompt. In the beginning the token count only includes the initial prompt.
            total_tokens = num_tokens_from_string(question_prompt, encoding="cl100k_base")

            # Perform retrieval based on the user's query
            results = vectordb_websupport_bot.similarity_search_with_score(restructured_query, k=50)
            print("len_results:" + str(len(results)))
            print("first result: " + str(results[0]))

            first_document_score = results[0][1]  # Extract the score of the first document

            # If cosine distance is below 0.3 continue answering question. Else hand off to support representative.
            if first_document_score < 0.3:
                results = adjust_similarity_scores(results, question=restructured_query,
                                                   word_intent_dict=words_to_check, multiplier=0.8)
                for doc in results:
                    doc = doc[0]
                    if doc.metadata.get("Source") == "webhelp-article":
                        link = doc.metadata.get("Link")
                        webhelp_article_content = doc.page_content
                        context = f" {webhelp_article_content}\nLink: {link}"
                        l_webhelp_articles.append(context)
                        print("l_webhelp_articles: " + str(l_webhelp_articles))

                    elif doc.metadata.get("Source") == "websupport question":
                        websupport_question = doc.page_content
                        websupport_answer = doc.metadata.get("Answer", "No answer found")
                        # Format the question and answer
                        context = f"Q: {websupport_question}\nA: {websupport_answer}"
                        l_webhelp_questions.append(context)

                    # Get the number of tokens
                    tokens = num_tokens_from_string(context, encoding="cl100k_base")

                    print("total tokens = " + str(total_tokens))

                    # If adding the answer would exceed the token limit, break out of the loop.
                    if total_tokens + tokens > 8000:
                        break
                    else:
                        total_tokens += tokens

                # Construct a prompt for GPT-3.5 Turbo based on the user's question
                current_prompt = question_prompt \
                    .replace('<<query>>', restructured_query) \
                    .replace('<<websupport_questions>>', "\n".join(l_webhelp_questions)) \
                    .replace("<<webhelp_article>>", " ".join(l_webhelp_articles))

                print("prompt: " + current_prompt)

                # Generate answer to prompt
                response = gpt3_1106_completion(prompt=current_prompt, log_directory=teams_bot_path + r"\gpt3_logs\prompt")
                print("response: " + response)

                # Add memory to retriever
                # Count how many tokens the retriever prompt has
                tokens_retriever = num_tokens_from_string(retriever_prompt, encoding="cl100k_base")
                if tokens_retriever > 3000:  # If token limit is reached, delete latest conversation turn
                    l_retriever_history = l_retriever_history[1:]

                # Reverse list, so newest chat history is added to the top of prompt
                l_reversed_retriever_history = l_retriever_history[::-1]

                # Delete old history and add new history with the latest Q&A to prompt
                testing_retriever_prompt = remove_history(retriever_prompt,
                                                          "<<newest message>>(.*?)<<oldest message>>",
                                                          "<<newest message>>\n<<history>>\n<<oldest message>>")

                updated_retriever_prompt = testing_retriever_prompt.replace('<<history>>',
                                                                            "\n".join(l_reversed_retriever_history))

                # Save history to retriever prompt.
                save_file(updated_retriever_prompt, teams_bot_path + r"\prompts\retriever_prompt.txt")

                # Add memory to Q&A prompt
                # In order not to exceed token length, only the last two conversation turns are added as history
                # to question_prompt.

                # Get last two conversation turns
                if len(l_qa_history) > 3:
                    l_qa_history = l_qa_history[1:]

                last_QA_pair = f"\nHuman: {restructured_query} \nAI: {response}"
                l_qa_history.append(last_QA_pair)

                print("prompt_history: " + " ".join(l_qa_history))
                # Delete all conversation turns
                question_prompt = remove_history(question_prompt,
                                                 "Human:(.*?)<<history>>",
                                                 "<<history>>")
                # Add new conversation turns
                question_prompt = question_prompt.replace('<<history>>', " ".join(l_qa_history) + "\n<<history>>")
                save_file(question_prompt, teams_bot_path + r"\prompts\question_prompt.txt")
                print("updated_history: " + question_prompt)

                await turn_context.send_activity(MessageFactory.text(response))
                query_count += 1
                with open(query_count_file, 'w') as count_file:
                    count_file.write(str(query_count))

            else:
                print("Im in the else block")
                response = "Leider kann ich deine Frage nicht beantworten. " \
                           "Soll ich ein Websupport-Ticket mit deiner Frage eröffnen?"

                state["need_help"] = "yes"
                print(str(state["need_help"]))
                save_file(str(state), teams_bot_path + r"\state.txt")
                await turn_context.send_activity(MessageFactory.text(response))

        if state["need_help"] == "yes" and state["edit_ticket"] == "none" and follow_up_answer == "yes":
            print("im in helping part")
            yes_inputs = ["ja", "yes", "jawohl", "yep", "j", "ok"]
            no_inputs = ["nein", "ne", "no", "nö", "nope", "n"]

            user_input = str(turn_context.activity.text).lower()
            if any(choice in user_input for choice in yes_inputs):
                websupport_ticket = open_file(teams_bot_path + r"\restructured_query.txt")

                try:
                    #raise Exception("Forced error for testing")
                    print("creating mail")

                    outlook = win32.Dispatch("Outlook.Application")  # Starts Outlook application
                    new_email = outlook.CreateItem(0)  # Creates new email item
                    new_email.To = "mefabe7562@marksia.com"  # for testing purposes a temp. email address is used
                    new_email.Body = websupport_ticket  # body of email
                    new_email.Display(False)  # Displays the new email item
                    mail_setup = "yes"
                    if mail_setup == "yes":
                        response = "Ich habe ein Mail in Outlook erstellt mit deiner Anfrage."
                        await turn_context.send_activity(MessageFactory.text(response))
                        state["need_help"] = "no"
                        state["edit_ticket"] = "none"
                        save_file(str(state), teams_bot_path + r"\state.txt")
                    else:
                        print("mailsetup dindt work")


                except:
                    print("creating other mail")
                    response = "Das ist deine Nachricht: '" + websupport_ticket + "\nMöchtest du die Nachricht noch" \
                                                                                  " bearbeiten, bevor ein Ticket " \
                                                                                  "daraus erstellt wird?"
                    state["need_help"] = "yes"
                    state["edit_ticket"] = "question"
                    save_file(str(state), teams_bot_path + r"\state.txt")
                    await turn_context.send_activity(MessageFactory.text(response))

            if any(choice in user_input for choice in no_inputs):
                print("no input: " + user_input)
                response = "Falls noch nicht nachgeschaut, könnten die (Webhelp)[https://applsupport.hslu.ch/webhelp/]" \
                           "behilflich sein. Ansonsten erreichst du den Websupport unter websupport@hslu.ch[mailto:websupport@hslu.ch]"
                state["need_help"] = "no"
                state["edit_ticket"] = "none"
                save_file(str(state), teams_bot_path + r"\state.txt")
                await turn_context.send_activity(MessageFactory.text(response))

            if not any(choice in user_input for choice in yes_inputs) and not any(
                    choice in user_input for choice in no_inputs):
                # Code to execute if user_input is neither in yes_inputs nor in no_inputs
                response = "Bitte antworte mit 'Ja' oder 'Nein'."
                state["need_help"] = "yes"
                state["edit_ticket"] = "none"
                save_file(str(state), teams_bot_path + r"\state.txt")
                await turn_context.send_activity(MessageFactory.text(response))

        if state["need_help"] == "yes" and state["edit_ticket"] == "question" and follow_up_edit_answer == "yes":
            yes_inputs = ["ja", "yes", "jawohl", "yep", "j", "ok"]
            nope_inputs = ["nein", "ne", "no", "nö", "nope", "n"]
            edit_input = str(turn_context.activity.text).lower()
            print("im in follow up edit answer")
            if any(choice in edit_input for choice in yes_inputs):
                state["need_help"] = "yes"
                state["edit_ticket"] = "yes"
                save_file(str(state), teams_bot_path + r"\state.txt")
                print(state)
                await turn_context.send_activity(MessageFactory.text("Bitte gib die bearbeitete Nachricht ein:"))

            if any(choice in edit_input for choice in nope_inputs) and state["edit_ticket"] == "question":
                print("edit input:" + edit_input)
                restructured_query = open_file(teams_bot_path + r"\restructured_query.txt")
                send_email(restructured_query)
                state["need_help"] = "no"
                state["edit_ticket"] = "none"
                save_file(str(state), teams_bot_path + r"\state.txt")
                await turn_context.send_activity(
                    MessageFactory.text("Ich habe ein E-Mail mit deiner Frage an den Websupport geschickt."))

            if not any(choice in edit_input for choice in yes_inputs) and not any(
                    choice in edit_input for choice in nope_inputs):
                response = "Ungültige Eingabe. Bitte antworte mit 'Ja' oder 'Nein'."
                state["need_help"] = "yes"
                state["edit_ticket"] = "question"
                save_file(str(state), teams_bot_path + r"\state.txt")
                await turn_context.send_activity(MessageFactory.text(response))

        if state["need_help"] == "yes" and state["edit_ticket"] == "yes" and new_message_received == "yes":
            new_message = str(turn_context.activity.text)
            save_file(new_message, teams_bot_path + r"\restructured_query.txt")
            response = "Das ist deine Nachricht: '" + new_message + "'\nMöchtest du die Nachricht noch" \
                                                                    " bearbeiten, bevor ein Ticket " \
                                                                    "daraus erstellt wird?"
            state["need_help"] = "yes"
            state["edit_ticket"] = "question"
            save_file(str(state), teams_bot_path + r"\state.txt")
            await turn_context.send_activity(MessageFactory.text(response))