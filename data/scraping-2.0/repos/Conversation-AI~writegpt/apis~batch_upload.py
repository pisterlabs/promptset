import os
import threading
from datetime import datetime

import openai
from flask import Blueprint, request
from flask_jwt_extended import get_jwt_identity, jwt_required

from apis.demo_service import generate_output
from helpers.db import db
from models.batch_upload_status import BatchUploadStatus
from models.user import User

# set up OpenAI
openai.api_key = os.environ.get("OPENAI_API_KEY")

batch_upload_bp = Blueprint('batch_upload', __name__)

from firebase_admin import storage
import pandas as pd


def send_url_via_email(emailId, user_id, url):
    from customerio import APIClient, SendEmailRequest, CustomerIOException
    api = APIClient(os.environ.get("CUSTOMERIO_API_KEY"))

    body = """Hi {email},
    <br><br>
    A recent batch email writing job is finished. You can download the output CSV here:
    <br><br>
    {url}
    <br><br>
    Best,<br>
    WriteGPT""".format(
        email=emailId, url=url)

    request = SendEmailRequest(
        to=emailId,
        transactional_message_id="3",
        identifiers={
            "email": emailId
        },
        _from="kai@theconversationai.com",
        subject="Your WriteGPT Batch Job is finished",
        body=body
    )

    try:
        api.send_email(request)
    except CustomerIOException as e:
        print("error: ", e)
    print("Email Sent at : ", emailId)


def update_status_data_to_firebase_collection(docId, status, url="-", user=None):
    doc_ref = db.collection('batch_upload_status').document(docId)
    now = datetime.now()
    doc_ref.update({"updated_at": now,
                    "status": status, "url": url})
    # send link to email to user
    if status == "Success":
        print("Sending email.....")
        send_url_via_email(user.email, user.id, url)


def upload_file_to_firebase_storage(filename, docId, json_data, user):
    # Setting up the blob
    file_name = filename.split(".")[0]
    now = datetime.now()
    bucket = storage.bucket('writegpt-cai.appspot.com')  # storage bucket
    blob = bucket.blob(file_name + "-output" + str(now) + ".csv")

    df = pd.DataFrame.from_dict(json_data)
    # Upload the blob from the content of the byte.
    blob.upload_from_string(df.to_csv(index=False), content_type='text/csv')
    blob.make_public()
    url = blob.public_url
    print("generated file url", blob.public_url)
    print("Updating db status and url.......")
    update_status_data_to_firebase_collection(docId, "Success", url, user)


def save_status_data_to_firebase_collection(filename, user_id):
    doc_ref = db.collection('batch_upload_status').document()
    doc_id = doc_ref.id
    now = datetime.now()
    collection_bup = BatchUploadStatus(filename=filename, created_at=now, updated_at="",
                                       status="In Progress", url="-",
                                       user_id=user_id)
    collection_bup.save()
    return collection_bup.id


def reformat_knowledge_base(knowledge_base):
    formatted_knowledge_base = []

    knowledge_base = knowledge_base.replace("\r\n", "\n")
    knowledge_base_rows = knowledge_base.split("\n\n")

    for row in knowledge_base_rows:
        data = row.split(":")
        formatted_knowledge_base.append({"title": data[0], "description": data[1]})

    return formatted_knowledge_base


def generate_output_and_write_csv(**kwargs):
    df = kwargs.get('df', {})
    filename = kwargs.get('filename', '')
    doc_id = kwargs.get('doc_id', 0)
    user = kwargs.get('user', {})
    retry_count = kwargs.get('retry_count', 0)
    try:
        print("Running.......")
        data = []
        count = 1
        for index, row in df.iterrows():
            print("_______________________")
            print("Row Count ",count)
            # get info from row and parse it to api
            json_data = {
                "url": row["url"],
                "sender_info": row["sender_info"],
                "recipient_info": row["recipient_info"],
                "word_count": row["word_count"],
            }

            if row["search_on_google"] != '' and row["search_on_google"] is not None:
                if not isinstance(row["search_on_google"], bool):
                    search_on_google = row["search_on_google"].upper()
                else:
                    search_on_google = row["search_on_google"]
                json_data["search_on_google"] = search_on_google

            if row["prompt"] is not None or row["prompt"] != "":
                json_data["prompt"] = row["prompt"]

            if row["template"] is not None or row["template"] != "":
                json_data["template"] = row["template"]

            if row["knowledge_base"] is not None or row["knowledge_base"] != "":
                json_data["knowledge_base"] = reformat_knowledge_base(row["knowledge_base"])

            output = generate_output(json_data)
            json_data["output"] = output
            data.append(json_data)
            # print("generating another response from file.............")
            count = count + 1
        print("uploading file to firebase.....")
        upload_file_to_firebase_storage(filename, doc_id, data, user)
    except Exception as e:
        print("Failed....", str(e))
        update_status_data_to_firebase_collection(doc_id, "Failed", url="-", user=None)
        # if retry_count == 0:
        #     print("Retrying...")
        #     thread = threading.Thread(target=generate_output_and_write_csv, kwargs={
        #         'df': df, 'filename': filename, 'doc_id': doc_id, 'user': user, 'retry_count': 1})
        #     thread.start()


@batch_upload_bp.route('/fetch_upload_progress_report', methods=['GET'])
@jwt_required()
def fetch_upload_progress_report():
    docID = request.args.get("docID", None)
    userID = request.args.get("userID", False)

    if docID is not None:
        data = BatchUploadStatus.get_by_docId(docID)
        return {"data": data.to_dict()}

    if userID:
        userID = get_jwt_identity()
        user_data = BatchUploadStatus.get_by_userId(userID)
        return {"data": user_data}


@batch_upload_bp.route('/', methods=['POST'])
@jwt_required()
def parse_csv_recieve_output():
    print("here in batch upload")
    user_id = get_jwt_identity()
    user = User.get_by_id(user_id)
    print(user_id, "user id")
    print(user, "user")
    print(user.email, user.id, "user")
    # user_id = 12333
    # user = {"id": 12333, "email": "chandnigoyal01@gmail.com"}
    file = request.files['batch_csvfile']
    print(file, "file")
    if not file:
        return 'No file uploaded.', 400
    filename = file.filename
    if not file.filename.endswith('.csv'):
        return 'Invalid file type, please upload a CSV file.', 400
    df = pd.read_csv(file)
    doc_id = save_status_data_to_firebase_collection(filename, user_id)

    # try:
    thread = threading.Thread(target=generate_output_and_write_csv, kwargs={
        'df': df, 'filename': filename, 'doc_id': doc_id, 'user': user, 'retry_count': 0})
    thread.start()
    # except:
    #     thread = threading.Thread(target=generate_output_and_write_csv, kwargs={
    #         'df': df, 'filename': filename, 'doc_id': doc_id, 'user': user})
    #     thread.start()

    return {"status": "In Progress", "documentID": doc_id}
