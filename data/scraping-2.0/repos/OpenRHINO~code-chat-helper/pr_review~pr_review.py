import os
import hmac
import hashlib
import openai
import logging
import json
import uuid
import re
from functools import wraps
from flask import Flask, request, abort
from github import Github
from pymongo import MongoClient

app = Flask(__name__)

client = MongoClient('mongodb', 27017)
db = client['pr_review']
collection = db['review_comments_and_conversations']

# Custom JSON formatter
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "asctime": self.formatTime(record, self.datefmt),
            "levelname": record.levelname,
            "event_id": getattr(record, "event_id", "not set"),
            "repo": getattr(record, "repo", "not set"),
            "pr": getattr(record, "pr", "not set"),
            "message": record.getMessage(),
        }
        return json.dumps(log_entry)

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logHandler = logging.StreamHandler()
formatter = JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)

# Set up OpenAI API client
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Set up GitHub API client
gh = Github(os.environ.get("GITHUB_TOKEN"))

# Set up webhook secret
webhook_secret = os.environ.get("WEBHOOK_SECRET")

def validate_signature(request):
    signature = request.headers.get("X-Hub-Signature-256")
    if signature is None:
        return False

    sha_name, signature = signature.split("=")
    if sha_name != "sha256":
        return False

    mac = hmac.new(webhook_secret.encode(), msg=request.data, digestmod=hashlib.sha256)
    return hmac.compare_digest(mac.hexdigest(), signature)


def attach_event_id_and_repo_pr(func):
    # Generate an event_id and attach it with repo name and pr number to the log record
    @wraps(func)
    def wrapper(*args, **kwargs):
        event_id = str(uuid.uuid4())
        event = request.get_json()
        pr = event["pull_request"]
        repo = event["repository"]

        logger = logging.getLogger()
        old_factory = logger.makeRecord

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.event_id = event_id
            record.repo = repo['full_name']
            record.pr = pr['number']
            return record

        logger.makeRecord = record_factory
        try:
            return func(*args, **kwargs, event_id=event_id)
        finally:
            logger.makeRecord = old_factory

    return wrapper

@app.route('/healthz')
def healthz():
    # 目前比较简单，后续可以添加任何需要的健康检查逻辑
    return "Healthy", 200

@app.route("/review_pr", methods=["POST"])
@attach_event_id_and_repo_pr
def review_pr(event_id):
    logger.info("Received user request")
    if not validate_signature(request):
        abort(401, "Invalid signature")
    logger.info("Webhook signature validated")

    event = request.get_json()

    logger.info(f"Webhook event type: {event['action']}")

    if event["action"] not in ["opened", "synchronize", "reopened"]:
        return "Ignoring non-PR opening/synchronize/reopening events", 200

    pr = event["pull_request"]
    repo = event["repository"]

    try:
        # Get the code changes from the PR
        logger.info(
            f"Fetching PR details from GitHub repo {repo['full_name']} #{pr['number']}"
        )
        gh_repo = gh.get_repo(repo["full_name"])
        gh_pr = gh_repo.get_pull(pr["number"])

        # Extract issue description from the PR body
        ref_numbers = re.findall(r"#(\d+)", gh_pr.body)
        issues_description = ""
        for ref_number in ref_numbers:
            issue_or_pr = gh_repo.get_issue(int(ref_number))
            if issue_or_pr.pull_request is None:  # This means it's an Issue
                issues_description += f"Issue #{ref_number}: {issue_or_pr.title}\n{issue_or_pr.body}\n\n"

        # Extract the code changes from the PR
        code_changes = []
        for file in gh_pr.get_files():
            full_file_content = gh_repo.get_contents(file.filename, ref=gh_pr.head.sha).decoded_content.decode()
            code_changes.append({
                "filename": file.filename,
                "patch": file.patch,
                "full_content": full_file_content
            })

    except Exception as e:
        logger.error(f"Error while fetching PR details from GitHub API: {e}")
        return "Error while fetching PR details from GitHub API", 500

    # Concatenate the changes into a single string
    logger.info("Preparing GPT request with code changes and context")
    changes_str = "Title: " + gh_pr.title + "\n"
    if gh_pr.body is not None:
        changes_str += "Body: " + gh_pr.body + "\n"
    if issues_description != "":
        changes_str += "---------------Issues referenced---------------\n"
        changes_str += issues_description
    for change in code_changes:
        changes_str += "---------------File changed---------------\n"
        changes_str += f"File: {change['filename']}\n\nPatch:\n{change['patch']}\n\nFull Content:\n{change['full_content']}\n"


    # Prepare the GPT prompt and store it in MongoDB
    messages = [
            {
                "role": "system",
                "content": 
"""
As an AI assistant with expertise in programming, your primary task is to review the pull request provided by the user.

When generating your review, adhere to the following template:
**[Changes]**: Summarize the main changes made in the pull request in less than 50 words.
**[Suggestions]**: Provide any suggestions or improvements for the code. Focus on code quality, logic, potential bugs and performance problems. Refrain from mentioning document-related suggestions such as "I suggest adding some comments", etc.
**[Clarifications]**: (Optional) If there are parts of the pull request that are unclear or lack sufficient context, ask for clarification here. If not, this section can be omitted.
**[Conclusion]**: Conclude the review with an overall assessment.
**[Other]**: (Optional) If there are additional observations or notes, mention them here. If not, this section can be omitted.

The user may also engage in further discussions about the review. It is not necessary to use the template when discussing with the user.
""",
            },
            {
                "role": "user",
                "content": f"Review the following pull request. The patches are in standard `diff` format. Evaluate the pull request within the context of the referenced issues and full content of the code file(s).\n{changes_str}\n",
            },
        ]
    try:
        logger.info("Creating the document to store the review messages in MongoDB")
        collection.insert_one({"uuid": event_id, "messages": messages})
    except Exception as e:
        logger.error(f"Error while creating the document to store the review messages in MongoDB: {e}")
        return "Error while creating the document to store the review messages in MongoDB", 500

    try:
        # Call GPT to get the review result
        logger.info("Sending request to OpenAI API")
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=messages
        )
        logger.info("Received responses from OpenAI API")
    except Exception as e:
        logger.error(f"Error while calling OpenAI API: {e}")
        return "Error while calling OpenAI API", 500

    try:
        logger.info("Storing the review results in MongoDB")
        collection.update_one({"uuid": event_id}, {"$push": {"messages": response.choices[0]['message']}})
    except Exception as e:
        logger.error(f"Error while storing the review results in MongoDB {e}")
        return "Error while storing the review results in MongoDB", 500

    final_review = f"""**[AI Review]** This comment is generated by an AI model (GPT-4 Turbo) via **v2** prompt.\n\n{response.choices[0]['message']['content'].strip()}\n
**[Note]** 
The above AI review results are for reference only, please rely on human expert review results for the final conclusion.
Usually, AI is better at enhancing the quality of code snippets. However, it's essential for human experts to pay close attention to whether the modifications meet the overall requirements. Providing detailed information in the PR description helps the AI generate more specific and useful review results.
For further discussion with the AI Reviewer, please visit: http://8.210.154.109:32765/conversation?uuid={event_id}\n\n"""
    logger.info("Final review prepared")

    logger.info("Translating review to Chinese")
    translate_messages = [{"role": "user", "content": f"将下面内容翻译为中文:\n{final_review}"}]
    try:
        translated_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=translate_messages
        )
    except Exception as e:
        logger.error(f"Error while fetching PR details from GitHub API: {e}")
        return "Error while fetching PR details from GitHub API", 500
    logger.info("Translation completed")

    try:
        # Post the GPT result as a PR comment
        logger.info("Submitting PR review comment")
        gh_pr.create_issue_comment(
            final_review
            + "\n"
            + translated_response.choices[0]["message"]["content"].strip()
        )

        logger.info("PR review comment submitted")
    except Exception as e:
        logger.error(f"Error while submitting PR review comment: {e}")
        return "Error while submitting PR review comment", 500

    return "Review submitted", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
