from flask import Blueprint, jsonify, session
from extensions import openai_db

"""
ZMC STUDENT ASSISTANT - DEVELOPMENT HELPER MODULE

Module: development.py

Developer: Julius Mwangi
Contact:
    - Email: juliusmwasstech@gmail.com

---

Disclaimer:
This project is a solo endeavor, with Julius Mwangi leading all
development efforts. For inquiries, concerns, or collaboration requests
related to development-specific routes and functions, please direct them
to the provided contact email.

---

About

Welcome to the development playground of the ZMC Student Assistant -
the `development.py` module. This module houses temporary routes and
functions designed to aid in the development phase, crafted with
innovation by Julius Mwangi.

Developer Information

- Name: Julius Mwangi
- Contact:
  - Email: [juliusmwasstech@gmail.com]
            (mailto:juliusmwasstech@gmail.com)

Acknowledgments

Special thanks to the incredible ALX TEAM for their unwavering support
and guidance. Their influence has been instrumental in shaping my journey
as a software engineer, particularly in developing efficient tools for
the development phase.

---

Note to Developers:
Feel free to explore, contribute, or connect. Your insights and feedback,
especially regarding development-specific routes and functions, are
highly valued and appreciated!

Happy coding!
"""


# Create a Blueprint for development-related routes
development_blueprint = Blueprint('development_blueprint', __name__)


@development_blueprint.route("/wipe_database", methods=["GET"])
def wipe_databases():
    """
    Endpoint for wiping database collections. Drops collections
    related to user accounts, openai threads, and thread sequences.
    Clears the session.

    Returns:
        jsonify: JSON response indicating the success or failure of the
        wipe operation.
    """
    session.clear()
    try:
        # Drop database collections
        openai_db["user_account"].drop()
        openai_db["openai_threads"].drop()
        openai_db["thread_sequence"].drop()

        # Invoke db_collections_check()
        if db_collections_check():
            # Create a new testing account
            # configure_user_account()

            return jsonify({"message": "wipe success"})
        return jsonify({"message": "wipe failed"})
    except Exception as e:
        return jsonify({"message": f"Error: {e}"})


def configure_user_account():
    """
    Configure a testing user account by inserting a document into the
    "user_account" collection.

    Returns:
        None
    """
    # Connect to the MongoDB database
    collection = openai_db["user_account"]

    # Define the document
    document = {
        "user": "Admin",
        "email": "admin@gmail.com",
        "school": "testing school",
        "level_of_school": "master",
        "password": "password",
        "tokens": 100000,
        "accumulating_tokens": 0,
        "lock": False,
        "user_styling": {"font_size": 15, "font_family": "Gruppo",
                         "text_color": "#29ADB2",
                         "background_color": "#040D12"}
    }

    # Insert the document into the collection
    result = collection.insert_one(document)

    # Print the inserted document's ID
    print("Inserted document ID:", result.inserted_id)


def db_collections_check():
    """
    Check if specified collections exist in the MongoDB database.

    Returns:
        bool: True if all collections are successfully dropped, False
        otherwise.
    """
    # Collections to check
    collections_list = ["user_account", "openai_threads", "thread_sequence"]
    db_collections = openai_db.list_collection_names()
    print(str(db_collections))
    success = False

    for collection in collections_list:
        # Check if the collection has been dropped successfully
        if collection not in db_collections:
            success = True
            print(f"Collection '{collection}' dropped successfully.")
        else:
            success = False
            print(f"Failed to drop collection '{collection}'.")
            break
    return success


def refuse_further_registrations():
    """
    Check if the number of documents in the 'openai_threads' collection exceeds
    a certain limit.

    Returns:
        - True: If the document count is greater than or equal to 5, indicating
        that further registrations should be refused.
        - False: If the document count is less than 5, indicating that further
        registrations are allowed.
        - True: If an exception occurs during the process, with details printed
        to the console.
    """
    try:
        # Assuming openai_db is a pre-defined MongoDB client and database
        #   connection
        collection = openai_db["user_account"]

        # Count documents in the collection
        document_count = collection.count_documents({})
        print(f"users available = {document_count}")

        # Check if the document count exceeds the limit
        if document_count >= 3:
            return True
        else:
            return False
    except Exception as e:
        # Print the error details to the console
        print(f"refuse_further_registrations Error = {e}")
        return True
