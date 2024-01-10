from flask import Flask, request, jsonify
from utils.pdf_utils import pdf  # Make sure this is the correct import path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import openai


app = Flask(__name__)


def init_app(app):
    @app.route("/uploadATS", methods=["POST", "GET"])
    def process_ats():
        # print("i have started")
        if request.method == "POST":
            # Extract text from PDF
            file = request.files["file"]
            pdf_text = pdf(file)  # Assuming pdf function returns the extracted text
            print(pdf_text)
            # Get the job description text
            text_jd = request.form.get("text")
            # print(text_jd)
            # Calculate similarity
            documents = [text_jd, pdf_text]
            count_vectorizer = CountVectorizer()
            sparse_matrix = count_vectorizer.fit_transform(documents)
            df = pd.DataFrame(
                sparse_matrix.todense(),
                columns=count_vectorizer.get_feature_names_out(),
            )
            cos_sim = cosine_similarity(df, df)
            similarity_score = cos_sim[1][0]
            similarity_percentage = round(float(similarity_score), 4) * 100
            print(similarity_percentage)
            prompt = f"""
            I will provide you with a Job description text and a resume text. You will give atleast 5 points where the resume can be improved so that the cosine similarity between the resume and the job description increases.
            pdf_text:{pdf_text} 
            text_jd: {text_jd}
            The reponse should be in the following format:
            "Sugested improvements:
            1. 
            2.
            3.
            4.
            5.
            """  
            client = openai.Client(api_key="XXXXX")
  
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150
            )

            improvement_suggestions = response.choices[0].message.content
            print(improvement_suggestions)

            # Return the similarity score to the frontend
            response_data = {
            "similarity": similarity_percentage,
            "improvements": improvement_suggestions
            }
            return jsonify(response_data)
