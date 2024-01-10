import json
from flask import current_app
from flask_restful import Resource, reqparse
from services.DBService import DBService
from services.OpenAIService import OpenAIService
from services.WikipediaService import WikipediaService
from services.PineconeService import PineconeService
from services.HuggingFaceService import HuggingFaceService

class TrueOrFalseGeneration(Resource):
    def __init__(self):
        self.wikipedia_service = WikipediaService()
        self.db_service = DBService()
        self.openai_service = OpenAIService()  
        self.pinecone_service = PineconeService("bite")
        self.hugging_face_service = current_app.hugging_face_service


    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('topic', type=str, required=True, help='Education topic is required')
        data = parser.parse_args()
        
        # questions_json = {
        # "questions": [
        #     {"statement": "Janny my boy is a cool guy.", "is_true": True},
        #     {"statement": "Einstein invented the theory of relativity.", "is_true": True},
        #     {"statement": "Rock paper scissors is the coolest game ever.", "is_true": True}
        #     ]
        # }
        
        # return {'message': 'True or False questions generated successfully', 'content': questions_json["questions"]}, 201

        # Fetch Wikipedia page 
        summary = self.wikipedia_service.get_summary(data['topic'])
        if summary is None:
            return {'message': 'Page does not exist'}, 404

        prompt = 'Given the following summary, please create three true or false questions of the following form: {"questions":[{"statement":"...", "is_true":true ...},{"statement":"...", "is_true":false}]} Do not get cut off. Do not respond with anything besides the JSON.' + summary
        gpt_response = self.openai_service.generate_json(prompt, 200)
        questions_json = json.loads(gpt_response.choices[0].text)


        # Generate the vector for the set of questions
        try:
            content_string = ' '.join([q['statement'] for q in questions_json["questions"]])
            vector = self.hugging_face_service.generate_vector(content_string)

            content_db = self.db_service.save_content(questions_json, "trueorfalse")
            self.pinecone_service.upsert(content_db.id, vector)
        except Exception as e:
            print("TRUE/FALSE", e)
            return {'message': 'Error in generating True or False content', 'error': str(e)}, 500

        return {'message': 'True or False questions generated successfully', 'content': questions_json["questions"]}, 201
