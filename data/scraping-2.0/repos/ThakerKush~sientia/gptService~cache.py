from dotenv import load_dotenv
import os
load_dotenv('../.env')
import json
import time
print(os.getenv("OPENAI_API_KEY"))
from gptcache import Cache, Config
from gptcache.adapter import openai
from gptcache.adapter.api import init_similar_cache
from gptcache.embedding import Onnx
from gptcache.manager import manager_factory
from gptcache.processor.post import random_one
from gptcache.processor.pre import last_content
from gptcache.similarity_evaluation import OnnxModelEvaluation

encoder = Onnx()
onnx_evaluation = OnnxModelEvaluation()

cache_config = Config(similarity_threshold=0.75)

sqlite_faiss_data_manager_theory = manager_factory(
    "sqlite,faiss",
    data_dir="theory_cache",
    scalar_params={
        "sql_url": "sqlite:///./theory_cache.db",
        "table_name": "theory_cache",
    },
    vector_params={
        "dimension": encoder.dimension,
        "index_file_path": "./theory_cache_faiss.index",
    },
)
theory_cache = Cache()
init_similar_cache(
    cache_obj=theory_cache,
    pre_func=last_content,
    embedding=encoder,
    data_manager=sqlite_faiss_data_manager_theory,
    evaluation=onnx_evaluation,
    post_func=random_one,
    config=cache_config,
)
math_cache = Cache()

sqlite_faiss_data_manager_math = manager_factory(
    "sqlite,faiss",
    data_dir="math_cache",
    scalar_params={
        "sql_url": "sqlite:///./math_cache.db",
        "table_name": "math_cache",
    },
    vector_params={
        "dimension": encoder.dimension,
        "index_file_path": "./math_cache_faiss.index",
    },
)
init_similar_cache(
    cache_obj=math_cache,
    pre_func=last_content,
    embedding=encoder,
    data_manager=sqlite_faiss_data_manager_math,
    evaluation=onnx_evaluation,
    post_func=random_one,
    config=cache_config,
)

quiz_cache_math = Cache()

sqlite_faiss_data_manager_quiz_math = manager_factory(
    "sqlite,faiss",
    data_dir="quiz_math_cache",
    scalar_params={
        "sql_url": "sqlite:///./quiz_math_cache.db",
        "table_name": "quiz_math_cache",
    },
    vector_params={
        "dimension": encoder.dimension,
        "index_file_path": "./quiz_math_cache_faiss.index",
    },
)
init_similar_cache(
    cache_obj=quiz_cache_math,
    pre_func=last_content,
    embedding=encoder,
    data_manager=sqlite_faiss_data_manager_quiz_math,
    evaluation=onnx_evaluation,
    post_func=random_one,
    config=cache_config,
)

quiz_cache_theory = Cache()

sqlite_faiss_data_manager_quiz_theory = manager_factory(
    "sqlite,faiss",
    data_dir="quiz_theory_cache",
    scalar_params={
        "sql_url": "sqlite:///./quiz_theory_cache.db",
        "table_name": "quiz_theory_cache",
    },
    vector_params={
        "dimension": encoder.dimension,
        "index_file_path": "./quiz_theory_cache_faiss.index",
    },
)
init_similar_cache(
    cache_obj=quiz_cache_theory,
    pre_func=last_content,
    embedding=encoder,
    data_manager=sqlite_faiss_data_manager_quiz_theory,
    evaluation=onnx_evaluation,
    post_func=random_one,
    config=cache_config,
)


def response_text(openai_resp):
    return openai_resp['choices'][0]['message']['content']

def ask(type: str, content: str):
    if type == "THEORY":
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            cache_obj = theory_cache,
            messages=[
                {'role': 'system', 'content': """You are a helpful tutor for theory, you will help user to memorize the given content, you will summaries the content into easy to remember pointers and provide with acronyms to remember 
                 the content if possible ( try your hardest to give out a good and rememberable acronym that rhymes if it doesn't rhyme then don't bother )."""},
                {
                    'role': 'user',
                    'content': content
                }
            ],
        )
        return response_text(response)
    if type == "MATH":
        response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        cache_obj = math_cache,
        messages=[
            {'role': 'system', 'content': """You are a helpful tutor for math , you will figure out the solution to the user's query on your own and then explain the algorithm behind solving the question with the help of the input question
              if the input is not a question or a math related question your response will be "this is not a math related question please input a math related question"."""},
            {
                'role': 'user',
                'content': content
            }
        ],
    )
    return response_text(response)


def askQuiz(type: str, content: str):

    if type == "MATH":
        example_questions = {
            "title":"quiz on derivatives",
            "questions":[
            {
                "question": "What is the derivative of 2x^2 + 3x + 1?",
                "options": ["4x + 3", "4x + 2", "4x + 1", "3x + 1"],
                "correct_answer": 0,
            }
            ]
        }


        quiz = openai.ChatCompletion.create(
        model= 'gpt-3.5-turbo-1106',
        cache_obj = quiz_cache_math,
        response_format={ "type": 'json_object' },
        messages=[
                {
                    "role": "system",
                    "content": f"""you will form a quiz based on the given an input math question, the quiz should have a title and generate 2 to 3 questions of the same topic with their answer, first work out your own 
                    solutions to the questions and then answer, make sure the quiz is related to the same topic and of a medium difficulty, make sure you jumble the index of the correct_answer so 
                    it's not always the same for all the question (this is an integral step because if the correct_answer is the same index for all question the quiz becomes guessable) is not 
                    the same option for all the question, only respond with the quiz and the response should be in the json format, for example: {json.dumps(example_questions, indent=2)}, here, the response is an array of objects, where the 
                    first property is question, which is the question text, then you have an array of strings callled options, where all the available options are listed, and finally,the 'correctAnswer' 
                    property represents the index of the corrent answer in the options array.""",
                },
                {"role": "user", "content": content}

            ]

        )
        return response_text(quiz)
    
    if type == "THEORY":

        exampleQuestions = {
            "title": "quiz on wifi",
            "questions":[
          {
            "question": 'WWhich device broadcasts a wireless signal in a Wi-Fi network?',
            "options": ["Modem", "Smartphone", "Router", "Ethernet switch"],
            "correct_answer": 2,
          }]
        }
        quiz = openai.ChatCompletion.create(
          model= 'gpt-3.5-turbo-1106',
          cache_obj = quiz_cache_theory,
          response_format = { "type": 'json_object' },
          messages = [
            {
              "role": 'system',
              "content": f"""you will form a quiz based on the given an input theory, the quiz should have a title and generate 2 to 3 question of the same topic with their answer, first work out your own solutions to the questions and then answer,
             make sure the quiz is related to the same topic and of a medium difficulty, the format of the quiz should be of multiple choice, make sure you jumble the index of the correct_answer so it's not always the same for all the question 
             (this is an integral step because if the correct_answer is the same index for all question the quiz becomes guessable) is not the same option for all the question, only respond with the quiz and the response should be in the json format,
               for example:  {json.dumps(exampleQuestions, indent=2)}, here, the response is an array of objects, where the first property is question, which is the question text, then you have an array of strings callled options, where all the 
               available options are listed, and finally, the 'correctAnswer' property represents the index of the corrent answer in the options array.""",
            },
            { "role": 'user', "content": content },
          ],
        )
        return response_text(quiz)