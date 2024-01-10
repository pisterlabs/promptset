import openai
import time
import json
openai.api_key = 'your api key'

def get_completion(messages, model="gpt-3.5-turbo-0613", temperature=0):
    response = ''
    fun = [
    {
        "name": "grading_student_writing",
        "description": """You are an English teacher. 
        Generate fair feedback and reviews for student writing.
            Rate the writing in these aspects:
            - Content and Relevance
            - Organization and Structure
            - Grammar and Mechanics
            - Vocabulary and Word Choice
            - Clarity and Cohesion

            For each aspect, score it on a scale of 0 to 5. 
            The format should be "score/5". 
            Provide precise and concise explanations for your ratings. 
            Students may ask you to explain your ratings and comments.
            """,

        "parameters": {
            "type": "object",
            "description": "Comment and score for student writing.",
            "properties": {
                "comment": {
                    "type": "string",
                    "description": "Comment and score for student writing. Should have around 200 words"

                }
            }
        },
        "required": ["comment"]
    }
]
    except_waiting_time = 0.1
    while response == '':
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                request_timeout=50,
                # functions= fun,
                max_tokens = 100,
                function_call="auto"
            )
        except Exception as e:
            time.sleep(except_waiting_time)
            if except_waiting_time < 2:
                except_waiting_time *= 2
    # if response.choices[0].message.get("function_call"):
    #     chat_response = json.loads(response.choices[0].message['function_call']['arguments'])['comment']
    #     return chat_response
    return response.choices[0].message["content"]
