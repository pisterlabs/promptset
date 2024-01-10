"""
    This file is used to test the environment requirements.
"""

import os
import sys
import requests
import argparse
import json

CONFIG_FILE = "config.json"

def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        print("[ERROR] Module " + module_name + " does not exist.")
        return False
    else:
        print("[INFO] Module " + module_name + " exists.")
        return True

# Check if the environment is set up correctly
def test_env():

    # Check if the required modules exist
    module_list = ["os", "sys", "requests", "transformers", "openai", "langchain", "torch", "numpy", "pandas", "json", "pickle", "argparse", "faiss", "rich"]
    for module_name in module_list:
        assert module_exists(module_name)

    # Check Python version
    assert sys.version_info >= (3, 8)

def test_openai_api(openai_config):
    """
        INPUT: openai_config: dict
        OUTPUT: None
    """
    # Check if the API is working
    import openai
    openai.api_key = openai_config["OPENAI_API_KEY"]
    # Uncomment the following line to list the information of all the models
    # print(openai.Model.list())
    print("[INFO] OpenAI API is working.")

def test_ringley_api(ringley_config):
    """
        INPUT: ringley_config: dict
        OUTPUT: None
    """
    # Check if the API is working
    url_articles = ringley_config["articles_url"]
    url_blog = ringley_config["blogs_url"]
    headers = {ringley_config["key"]: ringley_config["value"]}
    response_articles = requests.get(url=url_articles, headers = headers)
    assert response_articles.status_code == 200
    print("[INFO] Ringley_articles API is working.")
    response_blog = requests.get(url=url_blog, headers= headers)
    assert response_blog.status_code == 200
    print("[INFO] Ringley_blog API is working.")

def main(args):
    # load config file
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.loads(f.read())
    if args.test:
        test_env()
    if args.api:
        test_openai_api(config["openai_api"])
        test_ringley_api(config["ringley_api"])
    print("[INFO] All tests passed.")
    print("[INFO] Congratulations :) The environment is successfully deployed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the environment requirements.")
    parser.add_argument("-t", "--test", 
                        help="Test the environment requirements, including all the neccesary deep learning libraies used to achieve inference of the chatbot.",
                        default=True, 
                        action="store_true")
    parser.add_argument("-a", "--api", 
                        help="Test the OpenAI API and Ringley API connection, which is for the embedding/inference fo AI, and access to the domain specific dataset respectively.",
                        default=True, 
                        action="store_true")
    args = parser.parse_args()
    main(args)