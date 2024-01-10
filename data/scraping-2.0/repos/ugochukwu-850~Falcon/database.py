import os
from pymongo import MongoClient

from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader

client_key = os.getenv("MONGODB_KEY")
client = MongoClient(client_key)
client = client.falcon

# init collections
users = client["users"]
question_cursor = client["questions"]
drafts_cursor = client["drafts"]
files = client["files"]

# init models


def serialize_Files(file, many=False):
    if many:
        response = []
        for x in file:
            response.append(serialize_Files(x))
        return response
    return {
        "class": file["class"],
        "name": file["name"],
        "teacher": file["teacher"] if "teacher" in file else "ugochukwuchizaramoku@gmail.com",
        "created": file["created"],
        "id": file["id"]
    }
# serializers


def serialize_user(user) -> dict:
    return{
        "id": str(user["_id"]),
        "name": user["name"],
        "email": user["email"],
        "profile_pic_url": user["profile_pic_url"],
        "total_questions": user["total_questions"],
        "total_files": user["total_files"],
        "total_drafts": user["total_drafts"],
        "credentials": user["credentials"]
    }


def serialize_Questions(question, many=False):
    if many:
        return [serialize_Questions(x) for x in question]
    return{
        "_id": str(question["_id"]),
        "title": question["title"],
        "created": question["created"],
        "question_id": question["question_id"],
        "link": f"https://docs.google.com/forms/d/{question['question_id']}/edit"
    }
def main():
    ...


if __name__ == "__main__":
    main()
