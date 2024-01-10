# tasks.py
from celery import Celery
from celery.result import AsyncResult
import pymongo
import openai
from bson import ObjectId
from flask import jsonify
from decouple import config

openaikey = config('OPENAI_API_KEY')

celery = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

        
@celery.task()
def generateqinfo(details):
    # Move MongoDB and OpenAI connections into the try block
    with pymongo.MongoClient("mongodb+srv://lazim:lazim@cluster0.inykpf1.mongodb.net/?retryWrites=true&w=majority") as client:
        openai.api_key = openaikey

        db = client["LeedCode"]
        collection = db["Employer"]
        examid = details['examId']
        userid = details['userId']
        user_id = ObjectId(userid)
        exam_id = ObjectId(examid)
        questions = list(collection.find({"_id": user_id, "exams.exam_id": exam_id}, {"exams.$": 1}))

        for question in questions[0]["exams"][0]["questions"]:
            system_msg = "you are an AI machine that provides suitable examples and constraints for a given coding question"
            user_msg = "provide examples and constraints in the format (examples: 'example', constraints: 'constraints') for the given programming question: " + question["text"]

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.1
            )

            content = response['choices'][0]['message']['content']
            sections = content.strip().split("Examples:")
            constraints = sections[1].strip().split("Constraints:")
            examples = constraints[0].strip()
            constr = constraints[1].strip()

            # Update the specific question in the MongoDB collection using the filtered positional operator
            result = collection.update_one(
                {
                    "_id": user_id,
                    "exams.exam_id": exam_id,
                    "exams.questions.question_id": question["question_id"]
                },
                {"$set": {
                    "exams.$[exam].questions.$[ques].examples": examples,
                    "exams.$[exam].questions.$[ques].constraints": constr
                }},
                array_filters=[
                    {"exam.exam_id": exam_id},
                    {"ques.question_id": question["question_id"]}
                ]
            )

            if result.acknowledged:
                print("Success")
            else:
                print("Error")

    


def check_status(task_id):
    try:
        task_result = AsyncResult(task_id, app=celery)
        if task_result.ready():
            return jsonify({'status': 'success', 'result': task_result.result}), 200
        elif task_result.state == 'PENDING':
            return jsonify({'status': 'pending'}), 202
        else:
            return jsonify({'status': 'in progress'}), 202
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500