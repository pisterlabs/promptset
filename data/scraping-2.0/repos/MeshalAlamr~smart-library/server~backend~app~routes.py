import base64
import os
import time
from io import BytesIO
from tempfile import NamedTemporaryFile

import openai
import requests
from backend.app import app
from dotenv import load_dotenv
from flask import render_template, request

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# time.sleep(10)
backend_url = os.getenv("BACKEND_API_ADDRESS", "localhost")
backend_url = f"http://{backend_url}:8000"


print("Waiting for backend server to start...")
retry_count = 0
max_retries = 30
retry_interval = 3

while retry_count < max_retries:
    try:
        response = requests.get(f"{backend_url}/status")
        response.raise_for_status()  # Raise an exception if the response status is an error (e.g., 404, 500)
        print("Backend server started successfully!")
        break
    except requests.RequestException as e:
        print(f"Error connecting to the backend server: {e}")
        retry_count += 1
        time.sleep(retry_interval)
else:
    print("Maximum retries exceeded. Failed to start the backend server.")


def librarian(query, summary, topic, sentiment):
    prompt_template = f"""You are a smart helpful librarian that will assist users in finding resources. 
    Given a user's query, and the most relevant resource's summary, topic, and sentiment, provide a response to the user's query.
    Be enthusiastic in your response and relate to the user's query. Mention that the resource is below, but do not provide anything.

    User Query: {query}
    Resource Summary: {summary}
    Resource Topic: {topic}
    Resource Sentiment: {sentiment}

    Response:
    """

    ChatML = [
        {
            "role": "system",
            "content": "You are a smart helpful librarian that will assist users in finding resources.",
        },
        {"role": "user", "content": prompt_template},
    ]

    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=ChatML)

    return completion.choices[0].message.content


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "input-form" in request.form:
            print("Processing document...")
            file = request.files["pdf"]
            if not file:
                return render_template("index.html")

            # Encode the file content as base64
            file_content = file.read()
            file_encoded = base64.b64encode(file_content).decode("utf-8")

            # Send the file to the backend server
            print("Sending file to backend server...")
            response = requests.post(
                    f"{backend_url}/process", json={"file": file_encoded}
                )
            
            if response.status_code == 200:
                data = response.json()["data"]
                summary = data["summary"]
                topics = data["topics"]
                sentiment = data["sentiment"]
            # print("Extracting text from PDF...")
            # response = requests.post(
            #     f"{backend_url}/extract", json={"file": file_encoded}
            # )
            # if response.status_code == 200:
            #     extracted_text = response.json()["data"]
            # else:
            #     return render_template(
            #         "index.html", msg="Error: Could not extract text from PDF."
            #     )

            # print("Summarizing text...")
            # response = requests.post(
            #     f"{backend_url}/summarize", json={"text": extracted_text}
            # )
            # if response.status_code == 200:
            #     summary = response.json()["data"]

            # print("Predicting topics...")
            # response = requests.post(f"{backend_url}/topic", json={"text": summary})
            # if response.status_code == 200:
            #     topics = response.json()["data"]

            # print("Predicting sentiment...")
            # response = requests.post(f"{backend_url}/sentiment", json={"text": summary})
            # if response.status_code == 200:
            #     sentiment = response.json()["data"]

            print("Formatting data...")
            document = {
                "type": request.form["type"].title(),
                "name": request.form["name"],
                "author": request.form["author"],
                "year": request.form["year"],
                "publisher": request.form["publisher"],
                "summary": summary,
                "topics": topics,
                "sentiment": sentiment,
            }
            return render_template("index.html", document=document)

        elif "preview-form" in request.form:
            if "cancel-button" in request.form:
                return render_template(
                    "index.html", msg="Document not inserted. Operation cancelled."
                )
            else:
                document = {
                    "type": request.form["type"],
                    "name": request.form["name"],
                    "author": request.form["author"],
                    "year": request.form["year"],
                    "publisher": request.form["publisher"],
                    "summary": request.form["summary"],
                    "topics": request.form["topics"],
                    "sentiment": request.form["sentiment"],
                }
                response = requests.post(
                    f"{backend_url}/insert", json={"document": document}
                )
                if response.status_code == 200:
                    print("Document inserted successfully!")
                return render_template(
                    "index.html", msg="Document inserted successfully!"
                )
        else:
            return render_template("index.html")
    return render_template("index.html", document="")


@app.route("/users", methods=["POST", "GET"])
def users():
    if request.method == "POST":
        query = request.form["query"]
        print("Querying database...")
        response = requests.post(f"{backend_url}/search", json={"query": query})
        if response.status_code == 200:
            document = response.json()["data"]
        else:
            return render_template(
                "users.html",
                msg="An error occurred while querying the database. Please try again later.",
            )

        print("Sending query to librarian...")
        try:
            librarian_response = librarian(
                query, document["summary"], document["topics"], document["sentiment"]
            )
        except:
            print(
                "An error occurred while communicating with the Librarian. Please try again later."
            )
            librarian_response = ""

        return render_template(
            "users.html", document=document, answer=librarian_response
        )
    return render_template("users.html")


if __name__ == "__main__":
    app.run(debug=False)
