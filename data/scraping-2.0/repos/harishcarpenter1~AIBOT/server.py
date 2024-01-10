# Importing all required python libraries.
import os
import markdown
from flask import Flask, jsonify, request, render_template, make_response
from flask_cors import CORS
from flask.helpers import send_file
import git
import openai


# Create a Flask application instance
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) for the Flask app
CORS(app)
# Open AI API Key
openai.api_key = ""


# Read .java File Function from Temp Directory
def read_files_with_extension(directory, extension):
    # Initialize an empty list to store file contents
    file_contents = []
    # Check if the specified directory exists
    if os.path.exists(directory):
        # Iterate over the files in the directory
        for filename in os.listdir(directory):
            # Check if the file has the specified extension
            if filename.endswith(extension):
                # Construct the full file path
                file_path = os.path.join(directory, filename)
                # Open the file in read mode with utf-8 encoding
                with open(file_path, "r", encoding="utf-8") as file:
                    # Read the file contents and append to the list
                    file_contents.append(file.read())

    # Return the list of file contents
    return file_contents


# Clone the Repo and fetch .java files
def fetch_code_from_repository(repo_url, branch="main"):
    try:
        # Cloning the Gitub Repo
        repo = git.Repo.clone_from(repo_url, "temp_repo")
        repo.git.checkout(branch)
        # Reading .java files from Repository and returning
        java_code = read_files_with_extension("./temp_repo", ".java")
        return java_code

    # Exceptions May Occur
    except git.exc.GitCommandError as e:
        error_message = str(e)
        return (
            jsonify(
                message=f"Error while fetching Java code from the repository: {error_message}"
            ),
            500,
        )

    except Exception as e:
        error_message = str(e)
        return jsonify(message=f"An error occurred: {error_message}"), 500


# Model Implementation
def generate_feedback(prompt):
    # Using GPT-3.5-TURBO Model
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a code reviewer."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
    )
    # Remove trailing or leading whitspaces from response
    generated_text = response.choices[0].message.content.strip()
    # Return generated response
    return generated_text


# Mapping server url with inital feedback returning function
@app.route("/feedback", methods=["POST"])
def get_feedback():
    # Request Method is post
    if request.method == "POST":
        # Extracting the Repository Url from the Request
        data = request.json
        repository_url = data.get("url")

        # If Repo link empty
        if not repository_url:
            return jsonify(message="Please provide a GitHub repository URL"), 400

        # Function call to Fetch java code from Repository
        java_code = fetch_code_from_repository(repository_url)

        if isinstance(java_code, tuple) and len(java_code) == 2:
            # Error occurred while fetching Java code
            return java_code

        # Lists to store feedbacks
        feedback_list = []
        feedback_data = []

        # java_code = [ file1_code, file2_code ]
        # i --> index ( file no. ) and code ( .java code )
        # i --> 0 and code --> content of file1_code
        for i, code in enumerate(java_code):
            # For Each Java file Run the Prompt Command
            # Prompt command containig all the 10 guidelines mentioned in the document
            prompt = f"""
                Please provide a detailed review of the provided Java code based on the following guidelines:

                Guidelines for Java code reviews:

                1. Please provide feedback on the code's adherence to Java conventions.

                2. Replace imperative code with lambdas and streams: Identify any loops or verbose methods in the code and suggest replacing them with streams and lambdas for improved readability and functional programming. Provide examples of how to rewrite the code using streams and lambdas.

                3. Please make sure to avoid returning nulls if possible and handle potential null values appropriately, such as using null checks or utilizing the Optional class. Additionally, consider using annotations like @Nullable or @NonNull to catch null conflicts during code compilation.

                4. Please make sure that references exposed to the client code are not manipulated, even if the field is declared as final. Instead, consider cloning the reference or creating a new reference before assigning it to the field. Similarly, when returning references, avoid exposing internal mutable state.

                5. Handle exceptions with care: Analyze the exception handling in the code and ensure that catch blocks are ordered from most specific to least specific. Consider the use of checked exceptions for recoverable scenarios and handle exceptions appropriately.

                6. Please ensure that the appropriate data structure is chosen based on the specific requirements and characteristics of the data. Consider the pros and cons of different Java collections, such as Map, List, and Set.

                7. Review the access modifiers in the code and recommend keeping methods or variables private by default unless they need to be exposed to client code. Discuss the potential consequences of exposing certain methods or variables.

                8. Ensure interfaces are used instead of concrete implementations to allow for future flexibility and easier code maintenance.

                9. Assess the necessity and generic nature of interfaces used in the code. Recommend avoiding unnecessary interfaces that may require maintenance in the future. Explain when interfaces are beneficial and provide appropriate use cases.

                10. Override hashCode when overriding equals: Verify if the equals method is correctly overridden whenever necessary, especially for value objects. Emphasize the importance of overriding both equals and hashCode methods to ensure consistency and proper functionality.

                Please scan the Java code and clearly explain if any of the guidelines are not followed. Provide a detailed explanation of the specific guideline that is violated and suggest improvements for the code. 

                Code for Java file {i + 1}:
                {code}
            """

            # Generate the Feedback for ith index java file by passing it to model
            feedback = generate_feedback(prompt)
            # Store the feedback of ith java file in list
            feedback_list.append(feedback)

            # Append the object contains orignal java code and Feedback in HTML format
            feedback_data.append(
                {
                    "file_number": i + 1,
                    #  It is used to convert Markdown-formatted text into HTML
                    "feedback": markdown.markdown(feedback),
                    "code": code,
                }
            )

        # Render the template with feedback data
        template = render_template("feedback.html", feedback_data=feedback_data)

        # Write the content of template into feedback.html file
        with open("feedback.html", "w") as file:
            file.write(template)

        response = make_response(send_file("feedback.html", mimetype="text/html"))
        response.headers.set(
            "Content-Disposition", "attachment", filename="feedback.html"
        )
        # Return the response
        return response
    else:
        return jsonify(message="Method not allowed"), 405


# Run the script
if __name__ == "__main__":
    app.run(debug=True)
