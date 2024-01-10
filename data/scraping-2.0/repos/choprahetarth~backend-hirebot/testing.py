import csv
import os
import json
import string
import random
import openai
from bson import ObjectId
from flask import jsonify
from api.website_content_scrape import website_scrape
from api.scrape_resume import scrape_resume
from api.publications import Get_Published_Papers
from api.db import Authenticate
from api.langchain_util import generate_message, resume_evaluate_score
from flask import Flask, render_template, request, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FileField, RadioField, IntegerField
from wtforms.validators import DataRequired, NumberRange
from flask_pymongo import PyMongo, MongoClient
from instamojo_wrapper import Instamojo
from flask_cors import CORS

app = Flask(__name__)
CORS(app, support_credentials=True)
app.config[
    "MONGO_URI"] = "mongodb+srv://choprahetarth:45AJpXuKlK90Xc5s@cluster0.jcnnsrz.mongodb.net/hirebot.hirebot_user?retryWrites=true&w=majority"
app.config['SECRET_KEY'] = 'your-secret-key'
openai.api_key = "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"
mongo = PyMongo(app)

### 
authenticate_obj = Authenticate(
    'mongodb+srv://choprahetarth:45AJpXuKlK90Xc5s@cluster0.jcnnsrz.mongodb.net/?retryWrites=true&w=majority')

ALLOWED_EXTENSIONS = set(["pdf"])
app.config["UPLOAD_FOLDER"] = "upload"

client = MongoClient(
    'mongodb+srv://choprahetarth:45AJpXuKlK90Xc5s@cluster0.jcnnsrz.mongodb.net/?retryWrites=true&w=majority')  # replace with your MongoDB connection string
db = client['hirebot']  # replace with your database name
users = db['users']
payments = db["payments"]
coupons = db["coupons"]

API_KEY = '3ee066d067708f3bc08b68a14c9ad885'
AUTH_TOKEN = 'd1ba2e5687a8cf53b028b9d15abcf694'

api = Instamojo(api_key=API_KEY, auth_token=AUTH_TOKEN, endpoint='https://www.instamojo.com/api/1.1/')


class EmailForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired()])
    submit = SubmitField('Submit')


class OptionForm(FlaskForm):
    option = RadioField('Choose your path', choices=[('research', 'Research'), ('job', 'Job')])
    submit = SubmitField('Next')


# class EarlyAccess(FlaskForm):
#     option = StringField('Early Access Coupon Code', validators=[DataRequired()])
#     submit = SubmitField('Next')


class ResearchForm(FlaskForm):
    person_name = StringField('Name of Person', validators=[DataRequired()])
    gpt_option = RadioField('GPT Name', choices=[('gpt-4', 'GPT-4'), ('gpt-3.5-turbo', 'GPT-3.5-Turbo')])
    professor_website = StringField('Professor Website', validators=[DataRequired()])
    google_scholar = StringField('Google Scholar Link', validators=[DataRequired()])
    resume = FileField('Upload Resume', validators=[DataRequired()])
    submit = SubmitField('Submit')


class JobForm(FlaskForm):
    person_name = StringField('Name of Person', validators=[DataRequired()])
    gpt_option = RadioField('GPT Name', choices=[('gpt-4', 'GPT-4'), ('gpt-3.5-turbo', 'GPT-3.5-Turbo')])
    job_description = StringField('Job Description', validators=[DataRequired()])
    resume = FileField('Upload Resume', validators=[DataRequired()])
    temperature_setting = IntegerField('Temperature', validators=[NumberRange(min=0, max=2)])
    submit = SubmitField('Submit')


@app.route('/getemail', methods=['GET', 'POST'])
def get_email():
    form = EmailForm()
    if form.validate_on_submit():
        email = form.email.data
        return redirect(url_for('get_option', email=email))
    return render_template('email.html', form=form)


# @limiter.limit("1/minute") # change this part
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


@app.route('/option/<email>', methods=['GET', 'POST'])
def get_option(email):
    form = OptionForm()
    if form.validate_on_submit():
        option = form.option.data
        if option == 'research':
            return redirect(url_for('research_info', email=email))
        else:
            return redirect(url_for('job_info', email=email))
    return render_template('option.html', form=form)


@app.route('/research/<email>', methods=['GET', 'POST'])
def research_info(email):
    form = ResearchForm()
    if form.validate_on_submit():
        file_in_memory = form.resume.data
        data = {
            "email": email,
            "person_name": form.person_name.data,
            "gpt_option": form.gpt_option.data,
            "option": "research",
            "professor_website": form.professor_website.data,
            "google_scholar": form.google_scholar.data,
            "resume": scrape_resume(file_in_memory)
        }

        professors_website_text = read_professor_website(
            data["professor_website"])  # add a parameter here for summarization after GPT4 Access
        data["professors_website_text"] = professors_website_text

        publications = read_google_scholar(data["google_scholar"])
        data["publications"] = publications

        research_mail = generate_research_mail(data)

        data["research_mail"] = research_mail

        mongo.db.users.update_one(
            {"email": email},
            {"$push": {"submissions": data},
             "$inc": {"credits": -10}},  # Reduce credits by 10
            upsert=True
        )

        # save data to MongoDB
        return "Thanks for your submission!"
    return render_template('research.html', form=form)


# @app.route('/job/<email>', methods=['GET', 'POST'])
# def job_info(email):
#     form = JobForm()
#     if form.validate_on_submit():
#         file_in_memory = form.resume.data
#         data = {
#             "email": email,
#             "person_name": form.person_name.data,
#             "gpt_option": form.gpt_option.data,
#             "option": "job",
#             "job_description": form.job_description.data,
#             "resume": scrape_resume(file_in_memory),
#             "temperature_setting" : form.temperature_setting.data
#         }

#         linkedin_dm = generate_industry_linkedin_dm(data)
#         data["linkedin_dm"] = linkedin_dm

#         mongo.db.users.update_one(
#             {"email": email},
#             {"$push": {"submissions": data},
#              "$inc": {"credits": -10}},  # Reduce credits by 10
#             upsert=True
#         )

#         return jsonify(data["linkedin_dm"])

#     return render_template('job.html', form=form)
#     # return linkedin_dm


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    # check if the post request has the file part
    if "file" not in request.files:
        resp = jsonify({"message": "No file part in the request"})
        resp.status_code = 400
        return resp
    file = request.files["file"]
    if file.filename == "":
        resp = jsonify({"message": "No file selected for uploading"})
        resp.status_code = 400
        return resp
    if file and allowed_file(file.filename):
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], "resume.pdf"))
        # d = Pdf_Scrape("./Resume.pdf", "./resume.txt")
        # resume = d.scrape()
        # store this in the DB
        resp = jsonify({"message": "File successfully and read successfully"})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify({"message": "Allowed file types are pdf only."})
        resp.status_code = 400
        return resp


@app.route("/read_professor_website", methods=["GET"])
def read_professor_website(url):
    professors_website = website_scrape(url)
    professors_website_text = professors_website[
                              :6000
                              ]  # make a hard limit of 6000 characters
    # send this to database
    # Add a parameter here for summarization when we have the GPT-4 Access
    summarization = False
    if summarization:
        # get the response
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {
                    "role": "system",
                    "content": "You are provided the content of a professor's personal website. You need to summarize the content in a way that a student who is browsing the website can understand the main fields of study, projects and fields the professor is working in.",
                },
                {
                    "role": "user",
                    "content": """Here is the content of the professor's website - """
                               + professors_website_text,
                },
            ],
        )
        resp = response["choices"][0]["message"]["content"]
        professors_website_text = resp

    return professors_website_text


@app.route("/read_google_scholar", methods=["GET"])
def read_google_scholar(url):
    c = Get_Published_Papers(url)
    publications = c.extract_papers()
    # send this to database
    return publications


@app.route("/remove_coupon/<code>", methods=["GET"])
def delete_coupon_code(code):
    # Find and remove the coupon code
    result = coupons.delete_one({"Coupon": code})
    if result.deleted_count == 1:
        return "Coupon code removed successfully"
    else:
        return "Coupon code not found"


@app.route("/validate_coupon/<code>", methods=["GET"])
def check_coupon_code_exists(coupon_code):
    # Check if the coupon code exists in the collection
    count = coupons.count_documents({"Coupon": coupon_code})
    if count > 0:
        return True
    else:
        return False


@app.route("/authenticate", methods=["POST"])
def authenticate():
    event = json.loads(request.form.get('event'))
    email = json.loads(request.form.get('email'))
    status_code = authenticate_obj.handle_authentication(event, email)
    return status_code


@app.route("/generate_research_mail", methods=["POST"])
def generate_research_mail(data):
    # professors_website = json.loads(request.form.get('professors_website'))
    # publications = json.loads(request.form.get('publications'))
    # resume = json.loads(request.form.get('resume'))
    # gpt_name = json.loads(request.form.get('gpt_name'))

    resume = data["resume"]
    gpt_name = data["gpt_option"]
    professors_website = data["professors_website_text"]
    publications = data["publications"]  #### openai api key

    # get the response
    response = openai.ChatCompletion.create(
        model=gpt_name,
        messages=[
            {
                "role": "system",
                "content": "Act as a Graduate student requesting for a Research Assistantship with a professor while going for their master's degree via a personalized cold-email. The mail should be a personalized one, given the personal website of the professor and their latest publications.",
            },
            {
                "role": "user",
                "content": """Here is the content of the professor's website - """
                           + professors_website
                           + """. And here are his top 10 publications- """
                           + publications
                           + """ and here is my resume- """
                           + resume,
            },
        ],
    )
    resp = response["choices"][0]["message"]["content"]
    return resp


def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''
    for _ in range(length):
        random_index = random.randint(0, len(characters) - 1)
        random_string += characters[random_index]
    return random_string


@app.route("/generate_industry_linkedin_dm", methods=["POST"])
def generate_industry_linkedin_dm():
    email = request.form.get("email")
    job_description = str(request.form.get("job_description"))
    job_description = job_description[:1000]  # HARD LIMIT 1
    resume = request.files['resume']
    resume = str(scrape_resume(resume))
    resume = resume[:1000]  # HARD LIMIT 2
    name_of_referrer = request.form.get("person_name")
    gpt_name = request.form.get("gpt_option")
    temperature_setting = float(request.form.get("temperature_setting"))

    user = mongo.db.users.find_one(
        {"email": email},
    )

    if user["credits"] <= 0:
        response = {
            "error": "No credits left , Please pay to get credits",
            "error_code": "405"
        }
        return response

    data = {
        "email": email,
        "person_name": name_of_referrer,
        "gpt_option": gpt_name,
        "option": "job",
        "job_description": job_description,
        "resume": resume,
        "temperature_setting": temperature_setting
    }

    ## first response
    response = openai.ChatCompletion.create(
        model=gpt_name,
        messages=[
            {
                "role": "system",
                "content": f"Act as a Job Seeker requesting {name_of_referrer} a personalized referral for a job posting in the form of a LinkedIn DM. Make sure that the DM is precise and short, and emphasises how your resume is aligned with the job role, and keep length less than 250 words. Make sure to address it to {name_of_referrer} from the company mentioned in the job description.",
            },
            {
                "role": "user",
                "content": """Here is the content of my resume - """
                           + resume
                           + """. And here is the job description- """
                           + job_description
                           + """. Finally, Pick out the name of the candidate from the resume and address the message from their side.""",
            },
        ],
        temperature=temperature_setting
    )
    resp = response["choices"][0]["message"]["content"]
    ## second response
    heading_response = openai.ChatCompletion.create(
        model=gpt_name,
        messages=[
            {
                "role": "system",
                "content": "Act as a summarizer",
            },
            {
                "role": "user",
                "content": f"""From the linkedin message here -> {resp}, provide a 5 word summary. DO NOT WRITE MORE THAN FIVE WORDS."""
            },
        ],
        temperature=0, max_tokens=20
    )
    heading = heading_response["choices"][0]["message"]["content"]
    message_id = generate_random_string(10)
    data["message_id"] = message_id
    data["linkedin_dm"] = resp
    data["heading"] = heading
    mongo.db.users.update_one(
        {"email": email},
        {"$push": {"submissions": data},
         "$inc": {"credits": -1}},  # Reduce credits by 1
        upsert=True)

    return resp


@app.route('/update_message', methods=['POST'])
def update_message():
    user_email = request.form['email']
    message_id = request.form['message_id']
    message = request.form['message']
    user = users.find_one({'email': user_email})

    if not user:

        return "FAILURE"
    else:
        users.update_one({"submissions.message_id": message_id},
                         {"$set": {"submissions.$.linkedin_dm": message}})

    return "SUCCESS"


@app.route('/pay', methods=['POST'])
def pay():
    user_email = request.form['email']
    amount = request.form['amount']
    user = users.find_one({'email': user_email})

    if not user:
        # create a new user if doesn't exist
        user_id = users.insert_one({'email': user_email, 'credits': 0}).inserted_id
    else:
        user_id = user['_id']

    response = api.payment_request_create(
        amount=amount,
        purpose='Adding credits',
        send_email=False,
        email=user_email,
        redirect_url=url_for('handle_redirect', user_id=str(user_id), _external=True)
    )

    print(response)
    # redirecting to the payment page
    return response['payment_request']['longurl']


@app.route('/handle_redirect/<user_id>', methods=['GET'])
def handle_redirect(user_id):
    # Get payment id from request args
    payment_id = request.args.get('payment_id')

    # If payment id exists
    if payment_id:
        # Check if payment has already been processed
        existing_payment = payments.find_one({'_id': payment_id})
        if existing_payment:
            return render_template('error.html', message="This payment has already been processed.")

        # Make a GET request to Instamojo API to fetch payment details

        response = api.payment_detail(payment_id)
        print(response)
        amount = response['payment']['unit_price']
        status = response['payment']['status']

        # Check if payment was successful
        if status.lower() == 'credit':
            # Multiply the amount by 10 to calculate the credits
            # credits = int(float(amount) / 10)
            # add dynamic payments
            # hardcoding 100 credits
            credits = 100 
            print(credits)

            # Fetch the user from the database and update their credits
            user = users.find_one({'_id': ObjectId(user_id)})
            if user:
                users.update_one({'_id': ObjectId(user_id)}, {'$inc': {'credits': credits}})
                # Add payment to the database
                payments.insert_one(
                    {'_id': payment_id, 'user_id': ObjectId(user_id), 'amount': amount, 'credits': credits,
                     "buyer_email": response["payment"]["buyer_email"], "response": response["payment"]})

            # return render_template('success.html', amount=amount, credits=credits)
            return redirect("https://frontend-hirebot-zeta.vercel.app/dashboard", code=302)

        else:
            return render_template('failure.html')

    return render_template('failure.html')


@app.route('/credits', methods=['GET'])
def get_credits():
    user_email = request.args.get('email')
    user = users.find_one({'email': user_email})
    if user:
        return {'credits': user['credits']}
    else:
        return {'error': 'User not found'}, 404


@app.route('/create_user', methods=['GET'])
def create_user():
    user_email = request.args.get('email')
    coupon_code = request.args.get('coupon')

    user = users.find_one({'email': user_email})

    if not user:
        user_id = users.insert_one({'email': user_email, 'credits': 0}).inserted_id

        mongo.db.users.update_one(
            {"email": user_email},
            {"$inc": {"credits": 2}},  # Add 2 credits by default
            upsert=True
        )

    else:
        user_id = user['_id']

    if check_coupon_code_exists(coupon_code):
        mongo.db.users.update_one(
            {"email": user_email},
            {"$inc": {"credits": 5}},  # Add credits by 5
            upsert=True
        )
    delete_coupon_code(coupon_code)

    return str(user_id)


@app.route('/get_generated_dms', methods=['GET'])
def get_generated_dms():
    user_email = request.args.get('email')
    user = users.find_one({'email': user_email})
    if not user:
        return ""

    if  user.get('submissions'):
        return user.get('submissions')
    else:
        return {}


@app.route('/evaluate_resume', methods=['POST'])
def evaluate_resume():
    job_description = str(request.form.get("job_description"))
    job_description = job_description[:1000]  # HARD LIMIT 1
    resume = request.files['resume']
    resume = str(scrape_resume(resume))
    resume = resume[:1000]  # HARD LIMIT 2

    try:
        evaluation, score = resume_evaluate_score(resume,job_description)

    except Exception as e:
        error_message = f"An error occurred"
        return error_message, 500

    response = jsonify(
        {
            "evaluation": evaluation,
            "score": score
        }
    )

    response.status_code = 200

    return response

if __name__ == "__main__":
    app.run()
