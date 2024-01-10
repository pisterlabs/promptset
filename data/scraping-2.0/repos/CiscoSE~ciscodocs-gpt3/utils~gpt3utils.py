import json
import utils
import openai

def getApiKey(config_location):
    try:
        with open(config_location, "r") as file:
            config = json.loads(file.read())
            return config["api_key"]
    except:
        return None


def uploadFile(file, api_key):
    try:
        openai.api_key = api_key
        response = openai.File.create(file = open(file, encoding="utf8"), purpose="answers")
        return response["id"]
    except:
        return "Upload failed!"


def listFiles(api_key):
    openai.api_key = api_key
    return openai.File.list()


def deleteFile(file_id, api_key):
    try:
        openai.api_key = api_key
        response = openai.File(file_id).delete()
        return response
    except:
        return "File deletion failed!"


def deleteAllFiles(api_key):
    openai.api_key = api_key
    data = listFiles(api_key)
    for file in data["data"]:
        deleteFile(file["id"], api_key)
    return "All files deleted"


def answerQuestion(question_text, file_id, api_key):
    try:
        openai.api_key = api_key
        response = openai.Answer.create(
            search_model = "ada",
            model = "davinci",
            question = question_text,
            file = file_id,
            examples = [
                ["What is the latest version of DNA Center?", "The latest version is 2.1.2.4"],
                ["Please describe SWIM.", "SWIM is a feature that manages software upgrades and controls the consistency of image versions and configurations across your network."]
            ],
            examples_context = "Cisco DNA Center’s latest release, 2.1.2.4, is a major update to this solution, with enhancements that greatly facilitate SDA policy and segmentation, identification of network endpoints, Wi-Fi 6 upgrades, power-over-Ethernet (PoE) management, and security and ITSM integrations.",
            max_tokens = 100,
            max_rerank = 300
        )
        answer_text = response["answers"][0]
        return answer_text
    except:
        return "I do not know how to answer that."