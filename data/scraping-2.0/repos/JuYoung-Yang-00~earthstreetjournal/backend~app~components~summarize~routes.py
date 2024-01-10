# app/components/summarize/routes.py
import os
import sys
from dotenv import load_dotenv
load_dotenv()
import json
from flask import Blueprint, jsonify
from bson import ObjectId
from datetime import datetime
from app import mongo
from config import Config
from app.components.summarize.summarize import OpenAISummarizer

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

summary_bp = Blueprint('summary', __name__)

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        elif isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)

def insert_new_summary_to_mongo(summary_data):
    try:
        insert_result = mongo.db.summaries.insert_one(summary_data)
        return insert_result.inserted_id
    except Exception as e:
        print(f"An error occurred while inserting the new summary: {e}")
        return None

summarize_all_key = os.getenv('SUMMARIZE_ALL_KEY')

@summary_bp.route('/all/<string:summarize_all_key>', methods=['GET']) 
def summarize_all_articles(summarize_all_key):
    if summarize_all_key != summarize_all_key:
        return jsonify({"success": False, "message": "Unauthorized access"}), 401

    try:
        summarizer = OpenAISummarizer(Config.OPENAI_API_KEY)

        while True:
            summarized_article_ids = [str(summary['_id']) for summary in mongo.db.summaries.find({}, {"_id": 1})]
            query = {'_id': {'$nin': summarized_article_ids}} if summarized_article_ids else {}

            article_to_summarize = mongo.db.articles.find_one(query)
            if not article_to_summarize:
                break  

            word_count = len(article_to_summarize.get('content', '').split())
            if word_count > 2000:
                continue  

            article_json_string = json.dumps(article_to_summarize, cls=JSONEncoder)
            summarized_content = summarizer.generate_summary(article_json_string)
            
            if not isinstance(summarized_content, str):
                print("The response from OpenAI is not in the expected format.")
                continue  

            summarized_data = json.loads(summarized_content)
            required_keys = ["_id", "category", "title", "author", "source", "content", "date", "link", "mainpoints"]
            if not all(key in summarized_data for key in required_keys):
                print("The summarized content does not contain all required keys.")
                continue  

            insert_result = insert_new_summary_to_mongo(summarized_data)
            if not insert_result:
                print("Failed to insert the summary into the database.")
                continue 

            print(f"Successfully inserted article with ID: {article_to_summarize['_id']}")
            print(json.dumps(summarized_data, indent=4, cls=JSONEncoder))  

        return jsonify({"success": True, "message": "All articles processed"}), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500