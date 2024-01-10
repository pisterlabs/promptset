
from langchain.chat_models import ChatOpenAI
import json



def question_with_history (question, chat_history):

    model_name = "gpt-3.5-turbo"
    chat_model = ChatOpenAI(model_name=model_name, temperature=0)

    chat_history_template = "The chat history from previous rounds as: "
    for idx, chat_history_per_round in enumerate(reversed(chat_history)):
        chat_history_template += f"Round {idx+1}: {chat_history_per_round[0]}\n"
    
    template = f"""
    Sometimes the question I ask is relating to the previous rounds of question and previous answer. 
    Please refer to the chat history to give an improved version of questions. 
    Noted that rounds of chat history is ranked from the the latest to oldest.
    Noted that if the provided question is clear enough and have no realtionships with the chathistory, please just directly output the question.
    
    Example 1. 
    Provided question: What is the meaning of 7th item in the list? 
    If the provided chat history really contains the 7th item such as "7. Establishing long-term relationships with selected key strategic supplier partners through our "Partner to Win" approach."
    You should output "What is the meaning of 'Establishing long-term relationships with selected key strategic supplier partners through our "Partner to Win" approach.'?" 
    instead of "What is the meaning of 7th item in the list?

    Example 2.
    What is our supply chain strategy associated with digital transformation?
    In this question, if the provided chat history does not contain any information relating to question, and this question is quite straightforward, you can just output the question directly.
    You can just output "What is our supply chain strategy associated with digital transformation?" directly.

    PLEASE JUST GIVE THE OPTIMISED QUESTION WITHOUT ANY OTHER TEXTS. You can also refer the examples to construct your thinking of improving the question.

    Now, I provided the question is {question} and the chat history is {chat_history_template}, please just give me an optimised question ONLY.
    """

    updated_question = chat_model.predict(template)
    print ('the updated quesion is ' , updated_question)

    return updated_question



def save_tuples_to_json(tuples_list, filename):
    # Convert list of tuples to list of lists
    lists_list = [list(t) for t in tuples_list]

    with open(filename, 'w') as file:
        json.dump(lists_list, file)

def load_tuples_from_json(filename):
    with open(filename, 'r') as file:
        lists_list = json.load(file)

    # Convert list of lists back to list of tuples
    tuples_list = [tuple(l) for l in lists_list]
    return tuples_list
