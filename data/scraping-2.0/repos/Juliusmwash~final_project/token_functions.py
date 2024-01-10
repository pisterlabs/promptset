from flask import session
from flask_login import current_user
from extensions import openai_db


"""
ZMC STUDENT ASSISTANT - TOKEN FUNCTIONS MODULE

Module: token_functions.py

Developer: Julius Mwangi
Contact:
    - Email: juliusmwasstech@gmail.com

---

Disclaimer:
This project is a solo endeavor, with Julius Mwangi leading all
development efforts. For inquiries, concerns, or collaboration requests
related to user token consumption, please direct them to the provided
contact email.

---

About

Welcome to the token tracking center of the ZMC Student Assistant - the
`token_functions.py` module. This module specializes in keeping meticulous
records of user token consumption, crafted with precision by Julius Mwangi.

Developer Information

- Name: Julius Mwangi
- Contact:
  - Email: [juliusmwasstech@gmail.com]
            (mailto:juliusmwasstech@gmail.com)

Acknowledgments

Special thanks to the incredible ALX TEAM for their unwavering support
and guidance. Their influence has been instrumental in shaping my journey
as a software engineer, particularly in developing efficient token
tracking functionalities.

---

Note to Developers:
Feel free to explore, contribute, or connect. Your insights and feedback,
especially regarding token-related operations, are highly valued and
appreciated!

Happy coding!
"""


def token_updating_func(current_tokens):
    """
    Updates user's token information in the database.

    Args:
    - current_tokens (str): The number of tokens to be deducted.

    Returns:
    - str: A message indicating the success or failure of the update.
    """
    try:
        accumulating_tokens = 0

        prev_thrd_rqst = session.get("previous_thread_request", 0)
        thrd_cntn_timer = session.get("thread_continuation_timer", 0)

        if prev_thrd_rqst:
            session["previous_thread_request"] = False
            accumulating_tokens = int(session.get("math_variable", 0))
        elif thrd_cntn_timer:
            session["thread_continuation_timer"] = False
            accumulating_tokens = 0

        # Connect to the database
        collection = openai_db["user_account"]
        remaining_tokens = None

        result = collection.find_one({"email": current_user.email})
        if result:
            total_tokens = int(result["tokens"])
            if not prev_thrd_rqst:
                if not thrd_cntn_timer:
                    accumulating_tokens = int(result["accumulating_tokens"])
            deduct_tokens = accumulating_tokens + int(current_tokens)
            value_to_update = None

            remaining_tokens = total_tokens - deduct_tokens
            if remaining_tokens > 0:
                value_to_update = {"tokens": remaining_tokens,
                                   "accumulating_tokens": deduct_tokens,
                                   "lock": False}
            else:
                value_to_update = {"tokens": remaining_tokens,
                                   "accumulating_tokens": deduct_tokens,
                                   "lock": True}

            session["user_tokens"] = remaining_tokens

            # Update document
            result = collection.update_one(
                    {"_id": result["_id"]}, {"$set": value_to_update})
            # Check if the update was successful
            if result.acknowledged and result.modified_count > 0:
                return "update successful"
            else:
                return "update failed"
    except Exception as e:
        print(f"token_udating_func Error = {e}")
        return "Update failed"


def calculate_old_thread_tokens(thread_id):
    """
    Calculate the total tokens used in the specified thread.

    Args:
    - thread_id (str): The ID of the thread.

    Returns:
    - tuple: A tuple containing the type of request and the tokens used.
    """
    try:
        # Connect to the database
        collection = openai_db["openai_threads"]

        # Variable to store all previous tokens used in the thread
        tokens_used = 0

        result = collection.find({"thread_id": thread_id})
        if result:
            for doc in result:
                tokens_used += int(doc["tokens_consumed"])

            session["previous_thread_request"] = True
            session["math_variable"] = tokens_used
            return "old_thread_request", tokens_used

        session["previous_thread_request"] = True
        session["UNIVERSAL_ERROR"] = True
        print("calculate_old_thread_tokens Triggered Universal Error")
        return "Error"

    except Exception as e:
        print(f"token_calculations_decider Error = {e} Universal Error")
        session["previous_thread_request"] = True
        session["UNIVERSAL_ERROR"] = True
        return "Error"


def get_remaining_user_tokens():
    """
    Get the remaining tokens for the current user.

    Returns:
    - int: The remaining tokens.
    """
    try:
        # Connect to the database
        collection = openai_db["user_account"]
        result = collection.find_one({"email": current_user.email})
        if result:
            tokens_remaining = int(result["tokens"])

            # Save this data for latter retrival
            session["user_tokens"] = tokens_remaining

            return tokens_remaining
        return 0
    except Exception as e:
        print(f"get_remaining_user_tokens Error = {e}")
        return 0
