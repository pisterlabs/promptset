import os
import openai
from dotenv import load_dotenv
from flask import Flask, jsonify, request

app = Flask(__name__)
# Load the environment variables from the .env file
load_dotenv()
# Set up OpenAI API credentials
api_key = os.getenv("API_KEY")
openai.api_key = api_key

# Define the default route
@app.route("/")
def index():
    return "Welcome to the API!"

# Define the /api route to handle POST requests
@app.route("/api", methods=["POST"])
def api():
    message = request.json.get("message")
    augmented_message = f"{message} - recommend 6 jobs for these skills without any description only the job titles and each in a new line"
    completion = openai.Completion.create(
        engine="text-davinci-003",
        prompt=augmented_message,
        max_tokens=100,
        n=6,  
        stop=None,  
        temperature=0.7, 
    )
    job_titles = completion.choices[0].text.strip().split("\n")
    response = {"job_titles": job_titles}
    return jsonify(response)

# Define the /jobs/links route to handle links for jobs requests
@app.route("/jobs/links", methods=["POST"])
def get_job_links():
    message = request.json.get("message")
    augmented_message = f"{message} - recommend 6 jobs for these skills without any description only the job titles and each in a new line"
    completion = openai.Completion.create(
        engine="text-davinci-003", prompt=augmented_message, max_tokens=100, n=6
    )
    generated_text = completion.choices[0].text.strip()
    job_titles = generated_text.split("\n")
    job_data = []
    for title in job_titles:
        url_job_title = "+".join(title.split())
        indeed_url = f"https://in.indeed.com/jobs?q={url_job_title}"
        job_dict = {
            "job_title": title,
            "job_link": indeed_url,
        }
        job_data.append(job_dict)
    return jsonify(job_data)

# Define the /jobs/skills route to handle skills required for jobs requests
@app.route("/jobs/skills", methods=["POST"])
def get_job_skills():
    message = request.json.get("message")
    augmented_message = f"{message} - recommend 6 jobs for these skills without any description only the job titles and each in a new line"
    completion = openai.Completion.create(
        engine="text-davinci-003", prompt=augmented_message, max_tokens=100, n=6
    )
    generated_text = completion.choices[0].text.strip()
    job_titles = generated_text.split("\n")
    job_data = []
    for title in job_titles:
        skills_prompt = f"What are the necessary and additional skills required for a {title}?"
        skills_completion = openai.Completion.create(
            engine="text-davinci-003", prompt=skills_prompt, max_tokens=100, n=1
        )
        skills_text = skills_completion.choices[0].text.strip()
        skills_list = skills_text.split("Necessary Skills:")
        necessary_skills = skills_list[1].strip().split("\n") if len(skills_list) > 1 else []
        necessary_skills = [skill for skill in necessary_skills if skill]
        job_dict = {
            "job_title": title,
            "necessary_skills": necessary_skills,
        }
        job_data.append(job_dict)
    return jsonify(job_data)

# Define the /jobs/salary route to handle expected salary range for jobs requests
@app.route("/jobs/salary", methods=["POST"])
def get_job_salary():
    message = request.json.get("message")
    augmented_message = f"{message} - recommend 6 jobs for these skills without any description only the job titles and each in a new line"
    completion = openai.Completion.create(
        engine="text-davinci-003", prompt=augmented_message, max_tokens=100, n=6
    )
    generated_text = completion.choices[0].text.strip()
    job_titles = generated_text.split("\n")
    job_data = []
    for title in job_titles:
        salary_prompt = f"Give upper limit and lower limit of salary required for a {title}?Give in this format - lowerlimitval - upperlimitval"
        salary_completion = openai.Completion.create(
            engine="text-davinci-003", prompt=salary_prompt, max_tokens=100, n=1
        )
        salary_text = salary_completion.choices[0].text.strip()
        job_dict = {
            "job_title": title,
            "salary_range": salary_text,
        }
        job_data.append(job_dict)
    return jsonify(job_data)

# Define the /resources/books route to handle book resources for jobs requests
@app.route("/resources/books", methods=["POST"])
def get_resource_books():
    message = request.json.get("message")
    augmented_message = f"{message} - recommend 6 jobs for these skills without any description only the job titles and each in a new line"
    completion = openai.Completion.create(
        engine="text-davinci-003", prompt=augmented_message, max_tokens=100, n=6
    )
    generated_text = completion.choices[0].text.strip()
    job_titles = generated_text.split("\n")
    job_data = []
    for title in job_titles:
        resourcebook_prompt = f"Give 5 best books required for a {title}?"
        resourcebook_completion = openai.Completion.create(
            engine="text-davinci-003", prompt=resourcebook_prompt, max_tokens=100, n=1
        )
        resourcebook_text = resourcebook_completion.choices[0].text.strip()
        book_titles = resourcebook_text.split("\n")
        resource_dict = {
            "job_title": title,
            "resource_books": book_titles,
        }
        job_data.append(resource_dict)
    return jsonify(job_data)

# Define the /resources/freecourses route to handle free youtube resources for jobs requests
@app.route("/resources/freecourses", methods=["POST"])
def get_resources_free():
    message = request.json.get("message")
    augmented_message = f"{message} - recommend 6 jobs for these skills without any description only the job titles and each in a new line"
    completion = openai.Completion.create(
        engine="text-davinci-003", prompt=augmented_message, max_tokens=100, n=6
    )
    generated_text = completion.choices[0].text.strip()
    job_titles = generated_text.split("\n")
    job_data = []
    for title in job_titles:
        url_job_title = "+".join(title.split())
        youtube_url = f"https://www.youtube.com/results?search_query=playlists+for+{url_job_title}"
        resource_dict = {
            "job_title": title,
            "youtube_url": youtube_url,
        }
        job_data.append(resource_dict)
    return jsonify(job_data)

# Define the /resources/paidcourses route to handle paid udemy courses resources for jobs requests
@app.route("/resources/paidcourses", methods=["POST"])
def get_resources_paid():
    message = request.json.get("message")
    augmented_message = f"{message} - recommend 6 jobs for these skills without any description only the job titles and each in a new line"
    response = openai.Completion.create(
        engine="text-davinci-003", prompt=augmented_message, max_tokens=100
    )
    generated_text = response.choices[0].text.strip()
    job_titles = generated_text.split("\n")
    udemy_links = []
    for title in job_titles:
        url_job_title = "+".join(title.split())
        udemy_url = f"https://www.udemy.com/courses/search/?q={url_job_title}"
        udemy_links.append({"job_title": title, "udemy_link": udemy_url})
    return jsonify(udemy_links)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)