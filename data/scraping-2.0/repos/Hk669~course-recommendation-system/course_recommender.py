from flask import Flask, request, jsonify
from bson import ObjectId
import pymongo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import datetime
from openai import OpenAI
import os 
from dotenv import load_dotenv
load_dotenv()

PASS = os.getenv('PASS')
USER = os.getenv('USER')


app = Flask(__name__)

#connection to the mongo client
client = pymongo.MongoClient(f"mongodb+srv://{USER}:{PASS}@cluster0.jrov4ez.mongodb.net/?retryWrites=true&w=majority")
db = client.test

collection = db['students']
Courses = db.courses

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        elif isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)

@app.route('/recommendations', methods=["GET"])
def generate_rec_and_learning_path():

    user_email = request.args.get('user_email')  
    user = collection.find_one({"email": user_email})

    if not user or "interests" not in user:
        return jsonify({"error": "User interests not found"}), 404

    user_interests = user["interests"]

    # Retrieve courses from the DB
    courses = list(Courses.find())

    all_texts = [user_interests] + [f"{course['title']} {course['description']}" for course in courses]
    all_texts_flattened = [item if isinstance(item, str) else ' '.join(item) for item in all_texts]

    all_texts_flattened = [str(item) for item in all_texts_flattened]
    # Vectorize user interests and courses using a single TfidfVectorizer
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_texts_flattened)

    # Calculate cosine similarities
    user_vector = vectors[0]
    course_vectors = vectors[1:]
    cosine_similarities = cosine_similarity(user_vector, course_vectors).flatten()
    # print("User Interests:", ' '.join(user_interests))

    # Print courses with their descriptions and cosine similarities
    # debug statements
    # for course, similarity in zip(courses, cosine_similarities):
    #     print(f"\nCourse: {course['title']}")
    #     print(f"Description: {course['description']}")
    #     print(f"Cosine Similarity: {similarity}")


    top_indices = cosine_similarities.argsort()[-5:][::-1]
    top_indices = [int(index) for index in top_indices]
    
    # Check if courses list is long enough
    if max(top_indices) >= len(courses):
        return jsonify({"error": "Not enough courses available"}), 500
    
    similarity_threshold = 0.2  

    # Get top courses with similarity above the threshold
    top_courses = [dict(course) for course, similarity in zip(courses, cosine_similarities) if similarity > similarity_threshold]

    # Convert ObjectId to string for JSON serialization
    for course in top_courses:
        course['_id'] = str(course['_id'])
        course['createdAt'] = course['createdAt'].isoformat()
        course['updatedAt'] = course['updatedAt'].isoformat()
    # Return recommended courses in JSON format
    return JSONEncoder().encode({"recommendedCourses": top_courses})


# @app.route('/learning_path',methods=["POST"])
# def generate_learning_path():

#     user_query = request.json.get('user_query')
#     if not user_query:
#         return jsonify({'error':'Query is not given'}),404

#     # prompt = f"You are a helpful assistant. Create a flowchart for the user's learning path: {', '.join(user_query)}"
#     client1 = OpenAI()
#     # client1.api_key = ""

#     # response = client1.chat.completions.create(
#     # model="gpt-3.5-turbo",
#     # messages=[
#     #     {"role": "system", "content": prompt},
#     #     # {"role": "user", "content": user_query},
#     # ]
#     # )

#     # Make a request to the Chat API
#     response = client1.Completion.create(
#         engine="text-davinci-002",
#         prompt=f"Create a flowchart for the user's learning path: {user_query}",
#         max_tokens=150
#     )

#     return jsonify({response['choices'][0]['text']}) 

if __name__ == '__main__':
    app.run(debug=True)