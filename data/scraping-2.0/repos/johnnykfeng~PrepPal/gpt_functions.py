import openai
import json 
import os
import re
import pandas as pd

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# openai.api_key = os.environ['OPENAI_API_KEY']

def chat_completion(prompt,
                    model='gpt-3.5-turbo',
                    api_key=None,
                    system_prompt="You are a helpful assistant",
                    temperature=0.1,
                    top_p=1.0,
                    max_tokens=500):

    if api_key is not None:
      openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model=model,
        temperature = temperature,
        top_p=top_p,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}]
        )

    return response['choices'][0]['message']['content']

def spelling_finder(user_writing, chat_model="gpt-3.5-turbo"):
  # helper function, find spelling mistakes
  # works much better with gpt-4
  system_prompt = '''Given the following piece of writing, \
  find all the spelling mistakes in the text. \
  Output the mistakes as JSON with the following format:\
  {{"mistakes": ["<spelling mistake 1>", "<spelling mistake 2>", ...]\
    "correction":["<correct spelling 1>", "<correct spelling 2>, ...]}}'''

  user_prompt = f"{user_writing}"
  result = chat_completion(prompt=user_prompt, 
                           system_prompt=system_prompt, 
                           model=chat_model)
  try:
    mistakes_json = json.loads(result)
  except json.JSONDecodeError:
    print("JSONDecodeError")
    mistakes_json = result
  return mistakes_json

def find_word_positions(text, words):
  """find positions of words in text"""
  positions = []
  for word in words:
    p_start = text.index(word)
    p_end = p_start + len(word)
    mistake = text[p_start: p_end]
    print(mistake)
    positions.append((p_start, p_end))
  return positions

def wrap_words_in_text(text, word_list):
    for word in word_list:
        text = text.replace(word, f"**[{word}]**")
        # text = text.replace(word, f"<ins>{word}</ins>")
    return text

def contains_word(s, word):
    return re.search(f'\\b{word}\\b', s) is not None

def same_meaning(word1, word2, model ='gpt-3.5-turbo'):

  system = """You are a helpful assistant that helps students practice for their english proficieny exams. \
  Given the following two words enclosed in double square brackets [[]], \
  determine if they are synonyms or have the similar meaning. \
  Format the output as 'True' or 'False', nothing else. 'True' if they have the similar meaning, \
  'False' otherwise."""

  user_prompt = f"Two words to compare: [[{word1}]], [[{word2}]]"
  result = chat_completion(user_prompt, system_prompt = system, model = model)

  if contains_word(result, "True"):
    return True
  else:
    return False

def mc_questions_json(text, n=5):
  """Generate multiple choice questions based on the contents of the text.
  """
  system_prompt = """Given the corpus of text, \
  generate {n} multiple choice questions\
  based on the contents of the text. The goal of the these questions is to \
  quiz the reader. Make sure to randomize \
  the order of the answers for each question and evenly distribute the correct \
  answer across the options. Each question should be different and not repeated. \
  Format the questions in JSON as follows, make sure to use double quotes:\n \
  {{\
    "questions": [\
      {{\
        "question": "Who did X?",\
        "options": [\
         "A) Answer 1",\
         "B) Answer 2",\
         "C) Answer 3",\
         "D) Answer 4"
        ],\
        "correct_answer": "C) Answer 3", \
        "explanation": "Explanation of the correct answer" \
      }},\
      // More questions...\
    ]\
  }}
  """
  user_prompt = f"The text delimited in triple backticks:```{text}```"
  result = chat_completion(prompt=user_prompt,
                           system_prompt=system_prompt,
                           temperature=0)
  # result = chat_model(chat_prompt.format_prompt(n=n, text=text).to_messages())
  try:
    json_result = json.loads(result)
  except Exception as e:
    print(f"Error: {e}")
    json_result = result
  return json_result

def fitb_generate(reading_task, 
                  n=3, 
                  test_choice='ielts', 
                  model='gpt-3.5-turbo',
                  temperature = 0.4):
  """Generate fill in the blank exercises based on the contents of the text.

  Args:
      reading_task (str): long form text like a reading task.
      n (int, optional): Number of exercises to generate. Defaults to 3.
      test_choice (str, optional): _description_. Defaults to 'ielts'.
      model (str, optional): _description_. Defaults to 'gpt-3.5-turbo'.
      temperature (float, optional): _description_. Defaults to 0.4.

  Returns:
      _type_: _description_
  """
  
  system_prompt = f'''Given the reading task for {test_choice.upper()} \
  english proficiency test delimted in \
  triple backticks ```, generate {n} fill in the blank exercises \
  based on the contents of the text. The intent of these exercises \
  is to quiz the reader. Do not copy sentences from the given reading task, \
  create new sentence that relate to the content of the reading. \
  Format the output as a JSON file as follows:\
  [{{"incomplete_sentence": "<sentence with missing word as `___`>", "missing_word": \
  <missing word from the incomplete sentence>}}]
  '''
  user_prompt = f'Here is the reading task: ```{reading_task}```'
  result = chat_completion(prompt=user_prompt,
                            system_prompt=system_prompt,
                            model=model,
                            temperature=temperature)
  try:
    fitb = json.loads(result)
  except Exception as e:
    print("JSONDecodeError")
    return result

  return fitb

def get_writing_score(writing_text, task_question, test_choice="ielts"):
  """
  Calculate the writing score based on a given test type (IELTS or CELPIP).

  Parameters:
  - writing_text (str): The writing sample to be evaluated.
  - task_question (str): The specific question or task related to the writing test.
  - test_choice (str, optional): The type of the test being used for evaluation. Default is "ielts".
    Supported test types are "ielts" and "celpip".

  Returns:
  - result: The score or evaluation result for the provided writing test.

  Note:
  - If an unsupported test type is provided, the function will indicate "Invalid test choice".
  """

  # 2 different tests
  ielts_test = """
  You are a profesional IELTS writing task examer for General Training.
  Score the following text and provide subscore for each of the 4 IELTS criteria.Criterias:"Task achievement", \
  "Coherence and cohesion","Lexical resource","Grammatical range and accuracy".
  Writing task questions:\n{question_text}
  Writing task answer:\n{answer_text}"
  Output overall score and subscore in a dictionary format. Round the score to one decimal place with the first decimal digit only being 0 or 5.
  """
  celpip_test = """
  You are a professional CELPIP writing task examer.
  Score the following text and provide subscore for each of the 4 criteria.Criterias:"Content/Coherence", \
  "Vocabulary","Readability","Task Fulfillment".\
  Writing task questions:\n{question_text}
  Writing task answer:\n{answer_text}
  Output the overall score and subscore in a dictionary format. Round the score to integer.
  """

  # Switch between tests based on test_choice value
  if test_choice.lower() == "ielts":
      prompt_template = ielts_test
  elif test_choice.lower() == "celpip":
      prompt_template = celpip_test
  else:
      prompt_template = "Invalid test choice"

  # Apply the selected test, insert the writing text, and print the result
  scoring_prompt = prompt_template.format(question_text=task_question, answer_text=writing_text)
  # print(f"{scoring_prompt = }")
  result = chat_completion(scoring_prompt)

  try: # convert string to JSON
    result = json.loads(result)
  except JSONDecodeError:
    print("JSONDecoderError")

  return result


def grammar_judge(sentence, model = "gpt-3.5-turbo"):
  # herlper function, find grammar error.
    system_prompt = '''
                    Determine if the sentence has grammar mistakes. /
                    If the sentence has grammar mistakes, output 'True' , /
                    otherwise output 'False'. Only output true or false, nothing else.
                    '''
    user_prompt = f"{sentence}"
    result = chat_completion(prompt = user_prompt,
                             system_prompt = system_prompt,
                             model = model,
                             temperature=0)

    return result
  

def correct_sentence(sentence):
  # helper function, correct the sentences from grammar error.
    system_prompt = '''
                    You are an ai assistant that helps students \
                    correct and practice their english writing, \
                    the following sentence has grammar mistakes, \
                    correct them and output only the corrected sentence, nothing else.
                    '''
    user_prompt = f"{sentence}"
    result = chat_completion(prompt = user_prompt,
                             system_prompt = system_prompt,
                             model = "gpt-3.5-turbo",
                             temperature=0)

    return result
  
def find_sentence_positions(para, sentence):
  # helper function, help find the position of the sentences
    """find positions of words in text"""
    positions = []
    p_start = para.index(sentence)
    p_end = p_start + len(sentence)
    mistake = para[p_start: p_end]
    # print(mistake)
    positions.append((p_start, p_end))
    return positions
  
def full_grammar_corrector(text):
  # main function
  # split text into sentences
  # run the grammar_judge on each sentence
    sentence = text.split('\n')

    sentences = []
    sentence_lst = []
    for i in sentence:
        s = re.findall(r"[^.!?]+", i)
        sentences = sentences + s
    for s in sentences:
        if s != ' ':
            sentence_lst.append(s)

    df = []
    for s in sentence_lst:
        grammar = grammar_judge(s)
        if grammar == "True":
            corrected = correct_sentence(s)
            if corrected == s + ".":
                corrected = 'None'
                grammar = 'False'
        else:
            corrected = 'None'
            grammar = 'False'
        sentence_positions = find_sentence_positions(text, s)
        df.append({
                        'Sentences': s,
                        'Error': grammar,
                        'Corrected': corrected,
                        'Position': sentence_positions})
    return pd.DataFrame(df)
  
  
def create_suggestions(user_writing, model = "gpt-3.5-turbo"):
    system_prompt = '''
                    You are a professional IELTS writing examiner, \
                    provide suggestions for improving writing on the following paragraph \
                    based on the four criteria in IELTS writing, \
                    Task achievement, Coherence and cohesion, Lexical resource, and Grammatical range and accuracy. \
                    Output only the suggestions, and in four different paragraphs, each represent one criteria.
                    '''
    user_prompt = f"{user_writing}"
    lst = chat_completion(prompt = user_prompt, system_prompt = system_prompt, model = model, temperature=0)

    sentence = lst.split('\n\n')
    return sentence