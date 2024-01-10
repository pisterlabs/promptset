import promptStorage as prompts
import openai
import json
import utilityFunctions as util
import time
import asyncio



def main():
    pass

def find_and_replace_definitions(user_query):
    pass
    # Find relevant definition embeddings
    # Prompt gpt-4 to determine which definitions are most relevant
    # If there are multiple similar definitions, ask user to define which is most relevant
    # Reformat user_query with applicable definitions and return

def processing_stage(user_query):
    print("Starting processing stage...")
    # Get similar queries by calling GPT 3.5, maybe Google BARD instead
    similar_queries_list = []
    print("  - Converting query to list of questions using template")
    question_list = convert_query_to_question_list(user_query, used_model="gpt-3.5-turbo")
   
    print("  - Generating similar search queries for questions")
    similar_queries_list = get_similar_queries(question_list, user_query)
    return similar_queries_list, question_list

def convert_query_to_question_list(user_query, used_model):
    question_list = prompts.get_original_universal_answer_template(user_query)
    prompt_convert_question = prompts.get_prompt_convert_question(question_list)

    chat_completion =  util.create_chat_completion(used_model, api_key_choice="will", prompt_messages=prompt_convert_question, temp=0)
    converted_questions = chat_completion.choices[0].message.content
    converted_questions = converted_questions.split("\n")
    return converted_questions


def get_similar_queries(question_list, user_query):
    
    content_list = []
    lawful = prompts.get_prompt_similar_queries_lawful(user_query)
    #unlawful = prompts.get_prompt_similar_queries_unlawful(user_query)
    
    chat_completion = util.create_chat_completion(used_model="gpt-4", prompt_messages=lawful, temp=0, debug_print=True)
    lawful_result = chat_completion.choices[0].message.content
    #chat_completion = util.create_chat_completion(used_model="gpt-4", prompt_messages=unlawful, temp=0, debug_print=True)
    #unlawful_result = chat_completion.choices[0].message.content

    result_dct = json.loads(lawful_result)
    lawful = " ".join(result_dct["queries"])
    #result_dct = json.loads(unlawful_result)
    #unlawful = " ".join(result_dct["queries"])
    unlawful = None
        
    similar_queries_list = [lawful, lawful, lawful, unlawful, unlawful]
    return similar_queries_list

if __name__ == "__main__":
    main()