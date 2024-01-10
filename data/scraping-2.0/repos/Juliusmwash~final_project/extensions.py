from flask import Flask
from flask_login import LoginManager
from flask.sessions import SecureCookieSessionInterface
from openai import OpenAI
import os
from pymongo import MongoClient
from datetime import timedelta
from dotenv import load_dotenv

"""
ZMC STUDENT ASSISTANT - EXTENSIONS MODULE

Module: extensions.py

Developer: Julius Mwangi
Contact:
    - Email: juliusmwasstech@gmail.com

---

Disclaimer:
This project is a solo endeavor, with Julius Mwangi leading all
development efforts. For inquiries, concerns, or collaboration requests
related to app and MongoDB setup, please direct them to the provided
contact email.

---

About

Welcome to the core of the ZMC Student Assistant - the `extensions.py`
module. This module plays a pivotal role in setting up the Flask app
and connecting to the MongoDB database, meticulously crafted by
Julius Mwangi.

Developer Information

- Name: Julius Mwangi
- Contact:
  - Email: [juliusmwasstech@gmail.com]
            (mailto:juliusmwasstech@gmail.com)

Acknowledgments

Special thanks to the incredible ALX TEAM for their unwavering support
and guidance. Their influence has been instrumental in shaping my journey
as a software engineer, particularly in developing robust app and
database setups.

---

Note to Developers:
Feel free to explore, contribute, or connect. Your expertise and feedback,
especially concerning app configuration and database connections, are
highly valued and appreciated!

Happy coding!
"""


# Load environment variables from config file
dotenv_path = os.path.join(os.path.dirname(__file__), 'config.env')
load_dotenv(dotenv_path)

# Create app
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = 'project_testing'
app.session_interface = SecureCookieSessionInterface()

# Set the default session duration to 20 hours
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)

# Set the duration for the "Remember Me" cookie to 7 days
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=7)


login_manager = LoginManager(app)
login_manager.login_view = "log_reg_endpoint"

# Set up the MongoDB connection
openai_uri = os.environ.get('OPENAI_ADMIN_DATABASE_URI')
client = MongoClient(openai_uri)
openai_db = client['openaiDB']


api_key = os.environ.get('MY_OPENAI_API_KEY')
openai_client = OpenAI(api_key=api_key,)

"""
login_manager = LoginManager(app)
login_manager.login_view = 'login_endpoint'
"""
