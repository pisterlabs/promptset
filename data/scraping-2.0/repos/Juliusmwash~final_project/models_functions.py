from flask_login import UserMixin
from extensions import openai_db


"""
ZMC STUDENT ASSISTANT - MODELS FUNCTIONS MODULE

Module: models_functions.py

Developer: Julius Mwangi
Contact:
    - Email: juliusmwasstech@gmail.com

---

Disclaimer:
This project is a solo endeavor, with Julius Mwangi leading all
development efforts. For inquiries, concerns, or collaboration requests
related to user model and fetch operations, please direct them to the
provided contact email.

---

About

Welcome to the core of the ZMC Student Assistant - the
`models_functions.py` module. This module is dedicated to the user model
and functions facilitating user fetch operations, meticulously crafted
by Julius Mwangi.

Developer Information

- Name: Julius Mwangi
- Contact:
  - Email: [juliusmwasstech@gmail.com]
            (mailto:juliusmwasstech@gmail.com)

Acknowledgments

Special thanks to the incredible ALX TEAM for their unwavering support
and guidance. Their influence has been instrumental in shaping my journey
as a software engineer, particularly in developing robust user model and
fetch functionality.

---

Note to Developers:
Feel free to explore, contribute, or connect. Your insights and feedback,
especially concerning user-related operations, are highly valued and
appreciated!

Happy coding!
"""


class UserDetails(UserMixin):
    """
    Represents user details for authentication and tracking purposes.

    Args:
    - email (str): User's email address.
    - password (str): User's password.
    - student_name (str): User's name.
    - user_styling (str): User's styling information.
    - tokens (int): User's available tokens.
    - accumulating_tokens (int): User's accumulating tokens.
    - lock (object): Lock object for synchronization.

    Attributes:
    - email (str): User's email address.
    - password (str): User's password.
    - student_name (str): User's name.
    - user_styling (str): User's styling information.
    - tokens (int): User's available tokens.
    - accumulating_tokens (int): User's accumulating tokens.
    - lock (object): Lock object for synchronization.

    Methods:
    - get_id(): Returns the email as the identifier for Flask-Login.
    """
    def __init__(self,
                 email, password, student_name, user_styling, tokens,
                 accumulating_tokens, lock):
        self.email = email
        self.password = password
        self.student_name = student_name
        self.user_styling = user_styling
        self.tokens = tokens
        self.accumulating_tokens = accumulating_tokens
        self.lock = lock

    def get_id(self):
        return str(self.email)  # Return the email as the identifier


def get_user_from_db(email=None):
    """
    Retrieves user details from the database based on the email.

    Args:
    - email (str, optional): User's email address. Defaults to None.

    Returns:
    - UserDetails or None: User details if found, None otherwise.
    """
    try:
        user = None
        user_data = {}
        collection = openai_db['user_account']
        if email:
            user_data = collection.find_one({"email": email})
        if user_data:
            user = UserDetails(
                    email=user_data['email'],
                    password=user_data['password'],
                    student_name=user_data['student_name'],
                    user_styling=user_data['user_styling'],
                    tokens=user_data['tokens'],
                    accumulating_tokens=user_data['accumulating_tokens'],
                    lock=user_data['lock'],
                )
        return user
    except Exception as e:
        print(f"get_user_from_db Error: {e}")

    return None
