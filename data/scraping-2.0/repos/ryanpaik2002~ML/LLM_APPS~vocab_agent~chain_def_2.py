
# take in a word - make a call to the llm -> get a d
# get a mutlitple choice question, 
# get the answer, just the letter 



# get environmental 
import os
from dotenv import load_dotenv
import json

load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")



from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate, AIMessage
from langchain.chat_models import ChatOpenAI




# def process_multi_choice(list):
# word="adage"
def get_def_multi_choice_llm(word):
    # print(f"word: {word}")
    '''
    This function will query llm 
    input param: word:str
    return: langchain.schema.message.AIMessage
    '''
    q_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a helpful english vocab teacher, that helps make multiple choice questions"
                    "please provide just the multiple choice question"
                )       
            ),
            HumanMessagePromptTemplate.from_template("Can you make a multiple choice question asking for the definition for {word}, please give me 4 choices"),
            
            # HumanMessagePromptTemplate.from_template("What is the answer to {AIMessage[content]}?"),        
            ]
    )
    
    llm = ChatOpenAI()
    raw_question = llm(q_template.format_messages(word=word))
    # return raw_question
    print(f'raw_question: {raw_question}')
    dict_question = multi_choice_to_dict(raw_question)
    print(f'dict_question: {dict_question}')
    return dict_question
    
    
    # json_question=json.dumps(dict_question)
    # return json_question
    
    
    
def get_def_answer(word, result):
    """
    Queries OpenAI for the answer to the multiple choice question
    
    """
    A_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a helpful english vocab teacher, that helps make multiple choice questions"
                    "please provide just the letter to the correct answer"
                )       
            ),
            HumanMessagePromptTemplate.from_template("Can you give me the answer for the question asking for the correct definition of {word}: {result}"),
            HumanMessagePromptTemplate.from_template("please make sure your answer is a viable choice from a selection of A,B,C,D")
            # HumanMessagePromptTemplate.from_template("What is the answer to {AIMessage[content]}?"),
            # HumanMessagePromptTemplate.from_template("What is the answer to {AIMessage[content]}?"),        
            ]
    )
    
    llm = ChatOpenAI()
    answer = llm(A_template.format_messages(word=word, result=result))
    letter_answer = {"answer": answer.content[0]}
    
    return letter_answer


# process multiple choice questions (langchain object) into a dict
def multi_choice_to_dict(result):
    """
    formats the response from OpenAI for a multiple choice question for a single word
    """
    lines = result.content.split('\n')
    print(lines)
    question = lines[0]
    # Extract the answer choices (from the second line onward)
    answer_choices = lines[1:]
    # print(f"answer_choices: {answer_choices}")


    # convert answers choices to a dictionary
    # key is letter, values    
    choices_dict = {}
    for choice in answer_choices:
        if choice != '':
            key = choice[0]  # First letter as the key
            value = choice[3:]  # Sentence as the value (excluding the first three characters)
            choices_dict[key] = value
            # print(f"key: {key}, value: {value}")
    
    return question, choices_dict   # to return a dict
    # choices_json=json.dumps(choices_dict)
    # return choices_json
    




# word="adage"
# question = get_def_multi_choice_llm(word)
# print(question)

# answer = get_def_answer(question)
# print(answer)

# questions_dict = multi_choice_to_dict(question)
# print(questions_dict)
# print(question)





# q_template - Chat



