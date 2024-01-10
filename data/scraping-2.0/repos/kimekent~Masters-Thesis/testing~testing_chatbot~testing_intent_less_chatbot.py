"""
This testing chatbot is specifically designed for evaluating individual chatbot components.
It takes the following parameters: Test type, question, retriever, prompt, and input size.
-   The test type determines how many steps of the chatbot are executed before returning the results.
    To test the retriever only the retrieved documents are returned. No response is generated.
    To test the prompt and input size, documents are retrieved and in a next step these documents are added to the
    prompt and used to generate an answer.
-   The defined retriever, is a function that specifies which document embedding store is searched,
    what embedding model is used to embed the question, how the semantic similarity score is calculated,
    and how many documents are returned. For the two retrievers that are tested the semantic similarity
    is calculated using the dot product during the testing stage.
-   The prompt determines how the input into the test generation model is structured.
-   The input size determines how many documents are added to the prompt and fed as context to the text generation model.
-   To test memory, a multi-turn conversation is input into the model as a list of questions.
    Each question is related to the preceding one, necessitating the retention of prior generated responses in memory.
    A dedicated generation component manages this unique setup, storing previous responses in the prompt
    for reference.

To run this script update the 'path' variable to the root directory of this project and add your OpenAI API key to
'openaiapikey.txt' in the root directory of this project.
"""


# Import necessary modules and libraries
import openai
import os
import ast
from testing_functions import gpt3_1106_completion, open_file, num_tokens_from_string, remove_history, save_file, OpenAI_retriever


# 1. Setup-----------------------------------------------------------------------------------------

# Define variables and paths
path = r"C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\Masters-Thesis" # Change
testing_chatbot_path = path + r"\testing"
testing_prompt_logs_directory = testing_chatbot_path + r"\testing_chatbot\logs"

# Set the API key as an environment variable
os.environ["OPENAI_API_KEY"] = open_file(path + r"\openaiapikey.txt") # Add API Key to text file
openai.api_key = os.getenv("OPENAI_API_KEY")


# 2. Define chatbot----------------------------------------------------------------------------------

def main(test, retriever, query=None, max_input=None, question_prompt=None, retriever_prompt=None):
    if test == "retriever" or test == "prompt_max_input":
        print("Testing retriever or prompt_input_max")
        # In the retriever the document source, the question embedding model and the number of returned documents is already defined.
        retrieved_documents = retriever

        # To test retriever only the retrieved documents need to be returned.
        if test == "retriever":

            return retrieved_documents

        # To test prompt and input size, continue to text generation component.
        elif test == "prompt_max_input":
            print("Testing prompt_input_max")
            # Set up lists to store retrieved document info
            l_webhelp_articles = []  # This list will contain all retrieved webhelp articles retrieved to answer the question.
            l_webhelp_link = []  # This list will contain the urls to all retrieved webhelp articles.
            l_websupport_questions = []  # This list will contain the websupport questions and answers, retrieved to answer the question.
            l_websupport_id = []  # This list will contain the IDs of all websupport questions retrieved to answer the question.

            # This will keep track of the amount of input tokens.
            # Initialize with number of tokens in prompt, without retrieved context.
            total_tokens = num_tokens_from_string(question_prompt, encoding="cl100k_base")

            for doc in retrieved_documents:
                # If the retrieved document is a webhelp article append to the designated webhelp lists
                if doc['metadata']['source'] == "webhelp_article":
                    l_webhelp_link.append(doc["metadata"]["Link"])
                    context = doc['context']
                    l_webhelp_articles.append(context)
                # If the retrieved document is a websupport question append to the designated websupport lists
                elif doc['metadata']['source'] == "websupport_question":
                    l_websupport_id.append(doc["metadata"]["question_id"])
                    websupport_question = doc['context']
                    websupport_answer = doc['metadata']['answer']
                    # Format the question and answer
                    context = f"\nQ: {websupport_question}\nA: {websupport_answer}"
                    l_websupport_questions.append(context)

                # Get the number of tokens for the current context
                new_context_tokens = num_tokens_from_string(context, encoding="cl100k_base")

                # If adding the answer would exceed the token limit, break out of the loop
                if total_tokens + new_context_tokens > max_input:
                    break

                # If adding the new context tokens doesn't exceed max token length add context and continue to next doc.
                else:
                    total_tokens += new_context_tokens  # Update the total token count

            # Construct a prompt for GPT-3.5 Turbo based on the users question
            current_prompt = question_prompt\
                .replace("<<query>>", query) \
                .replace("<<websupport_questions>>", "\n".join(l_websupport_questions)) \
                .replace("<<webhelp_article>>", " ".join(l_webhelp_articles))

            response = gpt3_1106_completion(prompt=current_prompt, log_directory=testing_prompt_logs_directory,
                                            max_tokens=1000)
            if test == "prompt_max_input":
                return response, l_websupport_id, l_websupport_questions, l_webhelp_link, l_webhelp_articles


    elif test == "memory":
        print("Testing memory component")

        testing_retriever_prompt_logs = testing_chatbot_path + r"\testing_chatbot\logs\retriever"
        testing_retriever_prompt = remove_history(open_file(retriever_prompt),
                                                  "<<newest message>>(.*?)<<oldest message>>",
                                                  "<<newest message>>\n<<history>>\n<<oldest message>>")

        testing_question_prompt = remove_history(open_file(question_prompt),
                                                 "HISTORY:(.*?)<<history>>",
                                                 "HISTORY: <<history>>")

        # Set up lists and variables to store output over multiple conversation turns.
        query_count = 0
        l_webhelp_link_history_tracker = []
        l_websupport_id_history_tracker = []
        l_retriever_history = []
        l_qa_history = []  # This list will contain the previous questions and answers for the Q&A prompt.

        l_generated_responses = []
        l_restructured_queries = []

        for question in query:
            testing_retriever_prompt = testing_retriever_prompt
            testing_question_prompt = testing_question_prompt

            l_webhelp_articles = []  # This list will contain all retrieved webhelp articles retrieved to answer the question.
            l_webhelp_link = []  # This list will contain the urls to all retrieved webhelp articles.
            l_websupport_questions = []  # This list will contain the websupport questions and answers, retrieved to answer the question.
            l_websupport_id = []  # This list will contain the IDs of all websupport questions retrieved to answer the question.

            print("current query: " + str(question))

            # If query_count is 1 or more, the query is adjusted to include the subject from the first turn,
            # aiding the retriever in understanding the context for better document retrieval.
            # For query_count of 0, the original question is used unchanged.
            if query_count >= 1:
                # Add memory to retriever
                current_retriever_prompt = testing_retriever_prompt.replace('<<query>>', question)
                restructured_query = gpt3_1106_completion(prompt=current_retriever_prompt,
                                                          log_directory=testing_retriever_prompt_logs)

                print("Restructured query: " + restructured_query)
            else:
                restructured_query = question
            l_restructured_queries.append(restructured_query)

            # return top 'n' documents for the restructured query.
            words_to_check = ast.literal_eval(
                open_file(testing_chatbot_path + r"\testing_data\words_to_check_for_adjusted_similarity_score.txt"))
            retrieved_documents = OpenAI_retriever(query=restructured_query,
                                                   index_path=testing_chatbot_path + r"\testing_chatbot\indexes\OpenAI_index.json",
                                                   count=50,
                                                   word_intent_dict=words_to_check, multiplier=1.2)

            # This will keep track of the amount of input tokens. Initialize with number of tokens in prompt, without retrieved context.
            total_tokens = num_tokens_from_string(testing_question_prompt, encoding="cl100k_base")

            for doc in retrieved_documents:
                # If the retrieved document is a webhelp article append to the designated webhelp lists
                if doc['metadata']['source'] == "webhelp_article":
                    if doc["metadata"]["Link"] not in l_webhelp_link_history_tracker[:2]:
                        l_webhelp_link.append(doc["metadata"]["Link"])
                        context = doc['context']
                        l_webhelp_articles.append(context)
                    else:
                        continue

                # If the retrieved document is a websupport question append to the designated websupport lists
                elif doc['metadata']['source'] == "websupport_question":
                    if doc["metadata"]["question_id"] not in l_websupport_id_history_tracker[:2]:
                        l_websupport_id.append(doc["metadata"]["question_id"])
                        websupport_question = doc['context']
                        websupport_answer = doc['metadata']['answer']
                        # Format the question and answer
                        context = f"Q: {websupport_question}\nA: {websupport_answer}"
                        l_websupport_questions.append(context)
                    else:
                        continue

                # Get the number of tokens for the current context
                new_context_tokens = num_tokens_from_string(context, encoding="cl100k_base")

                # If adding the context would exceed the token limit, break out of the loop
                if total_tokens + new_context_tokens > max_input:
                    break

                # If adding the new context tokens doesn't exceed max token length add context and continue to next doc.
                else:
                    total_tokens += new_context_tokens  # Update the total token count

            # Construct a prompt for GPT-3.5 Turbo based on the users question
            current_prompt = testing_question_prompt \
                .replace("<<query>>", restructured_query) \
                .replace("<<websupport_questions>>", "\n".join(l_websupport_questions)) \
                .replace("<<webhelp_article>>", " ".join(l_webhelp_articles))
            print("current prompt:" + current_prompt)
            response = gpt3_1106_completion(prompt=current_prompt, log_directory=testing_prompt_logs_directory,
                                            max_tokens=1000)

            l_generated_responses.append(response)

            # Add output as memory
            # Create clean string to add to history part of the retriever prompt
            s_conversation_turn = "human: " + question + "\nAI: " + response
            l_retriever_history.append(s_conversation_turn)

            # Count how many tokens the retriever prompt has
            tokens_retriever = num_tokens_from_string(testing_retriever_prompt, encoding="cl100k_base")
            if tokens_retriever > 3000:  # If token limit is reached, delete latest conversation turn
                l_retriever_history = l_retriever_history[1:]

            # Reverse list, so newest chat history is added to the top of prompt
            l_reversed_retriever_history = l_retriever_history[::-1]

            # Delete old history and add new history with the latest Q&A to prompt
            testing_retriever_prompt = remove_history(testing_retriever_prompt,
                                                      "<<newest message>>(.*?)<<oldest message>>",
                                                      "<<newest message>>\n<<history>>\n<<oldest message>>")

            updated_retriever_prompt = testing_retriever_prompt.replace('<<history>>', "\n".join(l_reversed_retriever_history))

            # Save history to retriever prompt.
            save_file(updated_retriever_prompt, testing_chatbot_path + r"\testing_chatbot\testing_prompts\retriever\retriever_prompt.txt")

            # Add memory to Q&A prompt
            # In order not to exceed token length, only the last three conversation turns are added as history
            # to question_prompt.

            # Get last two conversation turns
            if len(l_qa_history) > 3:
                l_qa_history = l_qa_history[1:]

            last_QA_pair = f"\nHuman: {restructured_query} \nAI: {response}"
            l_qa_history.append(last_QA_pair)

            # Delete all conversation turns
            if query_count >= 1:
                testing_question_prompt = remove_history(testing_question_prompt,
                                                         "Human:(.*?)<<history>>",
                                                         "<<history>>")

            # Add new conversation turns
            updated_history = testing_question_prompt.replace('<<history>>', " ".join(l_qa_history) + "\n<<history>>")

            # Save question prompt with conversation history
            save_file(updated_history, testing_chatbot_path + r"\testing_chatbot\testing_prompts\answer_generation_with_history\qa_prompt_with_history.txt")

            query_count += 1 # This variables is used to decide if question need to be restructured

        return l_generated_responses, l_restructured_queries

if __name__ == "__main__":
    main()