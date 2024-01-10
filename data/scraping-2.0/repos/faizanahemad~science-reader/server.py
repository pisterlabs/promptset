import copy
import random
import secrets
import sys
from urllib.parse import unquote
from functools import wraps
import mmh3
import ast
import traceback
from flask import Flask, request, jsonify, send_file, session, redirect, url_for, render_template_string
from authlib.integrations.flask_client import OAuth
from flask_session import Session
from collections import defaultdict
import requests
from io import BytesIO
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.base import VectorStore
from collections import defaultdict
from Conversation import Conversation
from DocIndex import DocFAISS, DocIndex, create_immediate_document_index, ImmediateDocIndex
import os
import time
import multiprocessing
import glob
from rank_bm25 import BM25Okapi
from typing import List, Dict
from flask import Flask, Response, stream_with_context
import sys
sys.setrecursionlimit(sys.getrecursionlimit()*16)
import logging
import requests
from flask_caching import Cache
import argparse
from datetime import timedelta
import sqlite3
from sqlite3 import Error
from common import checkNoneOrEmpty, convert_http_to_https, DefaultDictQueue, convert_to_pdf_link_if_needed, \
    verify_openai_key_and_fetch_models, convert_doc_to_pdf
import spacy
from spacy.lang.en import English
from spacy.pipeline import Lemmatizer
from flask.json.provider import JSONProvider
from common import SetQueue
import secrets
import string
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import tiktoken
alphabet = string.ascii_letters + string.digits
import typing as t
# try:
#     import ujson as json
# except ImportError:
#     import json

import json

class FlaskJSONProvider(JSONProvider):
    def dumps(self, obj: t.Any, **kwargs: t.Any) -> str:
        """Serialize data as JSON.

        :param obj: The data to serialize.
        :param kwargs: May be passed to the underlying JSON library.
        """
        return json.dumps(obj, **kwargs)
    def loads(self, s: str, **kwargs: t.Any) -> t.Any:
        """Deserialize data as JSON.

        :param s: Text or UTF-8 bytes.
        :param kwargs: May be passed to the underlying JSON library.
        """
        return json.loads(s, **kwargs)
    
class OurFlask(Flask):
    json_provider_class = FlaskJSONProvider

os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None;
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    return conn

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

def create_tables():
    database = "{}/users.db".format(users_dir)

    sql_create_user_to_doc_id_table = """CREATE TABLE IF NOT EXISTS UserToDocId (
                                    user_email text,
                                    doc_id text,
                                    created_at text,
                                    updated_at text,
                                    doc_source_url text
                                ); """

    sql_create_user_to_votes_table = """CREATE TABLE IF NOT EXISTS UserToVotes (
                                    user_email text,
                                    question_id text,
                                    doc_id text,
                                    upvoted integer,
                                    downvoted integer,
                                    feedback_type text,
                                    feedback_items text,
                                    comments text,
                                    question_text text,
                                    created_at text,
                                    updated_at text
                                );"""
                                
    sql_create_user_to_conversation_id_table = """CREATE TABLE IF NOT EXISTS UserToConversationId (
                                    user_email text,
                                    conversation_id text,
                                    created_at text,
                                    updated_at text
                                ); """


    # create a database connection
    conn = create_connection(database)
    

    # create tables
    if conn is not None:
        # create UserToDocId table
        create_table(conn, sql_create_user_to_doc_id_table)
        # create UserToVotes table
        create_table(conn, sql_create_user_to_votes_table)
        create_table(conn, sql_create_user_to_conversation_id_table)
    else:
        print("Error! cannot create the database connection.")
        
    cur = conn.cursor()
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_UserToVotes_email_question ON UserToVotes (user_email, question_id)")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_User_email_doc_votes ON UserToVotes (user_email)")
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_UserToDocId_email_doc ON UserToDocId (user_email, doc_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_User_email_doc ON UserToDocId (user_email)")
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_UserToConversationId_email_doc ON UserToConversationId (user_email, conversation_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_User_email_doc_conversation ON UserToConversationId (user_email)")
    conn.commit()
        
        
from datetime import datetime

def addUserToDoc(user_email, doc_id, doc_source_url):
    conn = create_connection("{}/users.db".format(users_dir))
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO UserToDocId
        (user_email, doc_id, created_at, updated_at, doc_source_url)
        VALUES(?,?,?,?,?)
        """, 
        (user_email, doc_id, datetime.now(), datetime.now(), doc_source_url)
    )
    conn.commit()
    conn.close()
    
def addConversationToUser(user_email, conversation_id):
    conn = create_connection("{}/users.db".format(users_dir))
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO UserToConversationId
        (user_email, conversation_id, created_at, updated_at)
        VALUES(?,?,?,?)
        """, 
        (user_email, conversation_id, datetime.now(), datetime.now())
    )
    conn.commit()
    conn.close()


def getDocsForUser(user_email):
    conn = create_connection("{}/users.db".format(users_dir))
    cur = conn.cursor()
    cur.execute("SELECT * FROM UserToDocId WHERE user_email=?", (user_email,))
    rows = cur.fetchall()
    conn.close()
    return rows


def getCoversationsForUser(user_email):
    conn = create_connection("{}/users.db".format(users_dir))
    cur = conn.cursor()
    cur.execute("SELECT * FROM UserToConversationId WHERE user_email=?", (user_email,))
    rows = cur.fetchall()
    conn.close()
    return rows

def getAllCoversations():
    conn = create_connection("{}/users.db".format(users_dir))
    cur = conn.cursor()
    cur.execute("SELECT * FROM UserToConversationId")
    rows = cur.fetchall()
    conn.close()
    return rows

def addUpvoteOrDownvote(user_email, question_id, doc_id, upvote, downvote):
    assert not checkNoneOrEmpty(question_id)
    assert not checkNoneOrEmpty(user_email)
    assert not checkNoneOrEmpty(doc_id)
    conn = create_connection("{}/users.db".format(users_dir))
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO UserToVotes
        (user_email, question_id, doc_id, upvoted, downvoted, feedback_type, feedback_items, comments, question_text, created_at, updated_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)
        """, 
        (user_email, question_id, doc_id, upvote, downvote, None, None, None, None, datetime.now(), datetime.now())
    )
    conn.commit()
    conn.close()

def addGranularFeedback(user_email, question_id, feedback_type, feedback_items, comments, question_text):
    assert not checkNoneOrEmpty(question_id)
    assert not checkNoneOrEmpty(user_email)
    conn = create_connection("{}/users.db".format(users_dir))
    cur = conn.cursor()

    # Ensure that the user has already voted before updating the feedback
    cur.execute("SELECT 1 FROM UserToVotes WHERE user_email = ? AND question_id = ?", (user_email, question_id))
    if not cur.fetchone():
        raise ValueError("A vote must exist for the user and question before feedback can be provided")

    cur.execute(
        """
        UPDATE UserToVotes 
        SET feedback_type = ?, feedback_items = ?, comments = ?, question_text = ?, updated_at = ?
        WHERE user_email = ? AND question_id = ?
        """,
        (feedback_type, ','.join(feedback_items), comments, question_text, datetime.now(), user_email, question_id)
    )
    conn.commit()
    conn.close()



def getUpvotesDownvotesByUser(user_email):
    conn = create_connection("{}/users.db".format(users_dir))
    cur = conn.cursor()
    cur.execute("SELECT SUM(upvoted), SUM(downvoted) FROM UserToVotes WHERE user_email=? GROUP BY user_email ", (user_email,))
    rows = cur.fetchall()
    conn.close()
    return rows

def getUpvotesDownvotesByQuestionId(question_id):
    conn = create_connection("{}/users.db".format(users_dir))
    cur = conn.cursor()
    cur.execute("SELECT SUM(upvoted), SUM(downvoted) FROM UserToVotes WHERE question_id=? GROUP BY question_id", (question_id,))
    rows = cur.fetchall()
    conn.close()
    return rows

def getUpvotesDownvotesByQuestionIdAndUser(question_id, user_email):
    conn = create_connection("{}/users.db".format(users_dir))
    cur = conn.cursor()
    cur.execute("SELECT SUM(upvoted), SUM(downvoted) FROM UserToVotes WHERE question_id=? AND user_email=? GROUP BY question_id,user_email", (question_id, user_email,))
    rows = cur.fetchall()
    conn.close()
    return rows


def removeUserFromDoc(user_email, doc_id):
    conn = create_connection("{}/users.db".format(users_dir))
    cur = conn.cursor()
    cur.execute("DELETE FROM UserToDocId WHERE user_email=? AND doc_id=?", (user_email, doc_id,))
    conn.commit()
    conn.close()
    
def removeUserFromConversation(user_email, conversation_id):
    conn = create_connection("{}/users.db".format(users_dir))
    cur = conn.cursor()
    cur.execute("DELETE FROM UserToConversationId WHERE user_email=? AND conversation_id=?", (user_email, conversation_id,))
    conn.commit()
    conn.close()


def keyParser(session):
    keyStore = {
        "openAIKey": os.getenv("openAIKey", ''),
        "mathpixId": os.getenv("mathpixId", ''),
        "mathpixKey": os.getenv("mathpixKey", ''),
        "cohereKey": os.getenv("cohereKey", ''),
        "ai21Key": os.getenv("ai21Key", ''),
        "bingKey": os.getenv("bingKey", ''),
        "serpApiKey": os.getenv("serpApiKey", ''),
        "googleSearchApiKey":os.getenv("googleSearchApiKey", ''),
        "googleSearchCxId":os.getenv("googleSearchCxId", ''),
        "openai_models_list": os.getenv("openai_models_list", '[]'),
        "scrapingBrowserUrl": os.getenv("scrapingBrowserUrl", ''),
        "vllmUrl": os.getenv("vllmUrl", ''),
        "vllmLargeModelUrl": os.getenv("vllmLargeModelUrl", ''),
        "vllmSmallModelUrl": os.getenv("vllmSmallModelUrl", ''),
        "tgiUrl": os.getenv("tgiUrl", ''),
        "tgiLargeModelUrl": os.getenv("tgiLargeModelUrl", ''),
        "tgiSmallModelUrl": os.getenv("tgiSmallModelUrl", ''),
        "embeddingsUrl": os.getenv("embeddingsUrl", ''),
        "zenrows": os.getenv("zenrows", ''),
        "brightdataUrl": os.getenv("brightdataUrl", ''),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ''),
    }
    if keyStore["vllmUrl"].strip() != "" or keyStore["vllmLargeModelUrl"].strip() != "" or keyStore["vllmSmallModelUrl"].strip() != "":
        keyStore["openai_models_list"] = ast.literal_eval(keyStore["openai_models_list"])
    for k, v in keyStore.items():
        key = session.get(k, v)
        if key is None or (isinstance(key, str) and key.strip() == "") or (isinstance(key, list) and len(key) == 0):
            key = v
        if key is not None and ((isinstance(key, str) and len(key.strip())>0) or (isinstance(key, list) and len(key)>0)):
            keyStore[k] = key
        else:
            keyStore[k] = None
    return keyStore
    

def generate_ngrams(tokens, n):
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.ERROR,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.getcwd(), "log.txt"))
    ]
)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
log = logging.getLogger('faiss.loader')
log.setLevel(logging.ERROR)
logger.setLevel(logging.ERROR)
time_logger = logging.getLogger(__name__ + " | TIMING")
time_logger.setLevel(logging.INFO)  # Set log level for this logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', help='The folder where the DocIndex files are stored', required=False, default=None)
    parser.add_argument('--login_not_needed', help='Whether we use google login or not.', action="store_true")
    args = parser.parse_args()
    login_not_needed = args.login_not_needed
    folder = args.folder
    
    if not args.folder:
        folder = "storage"
else:
    folder = "storage"
    login_not_needed = True
app = OurFlask(__name__)
app.config['SESSION_PERMANENT'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SESSION_TYPE'] = 'filesystem'
app.config["GOOGLE_CLIENT_ID"] = os.environ.get("GOOGLE_CLIENT_ID")
app.config["GOOGLE_CLIENT_SECRET"] = os.environ.get("GOOGLE_CLIENT_SECRET")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")
app.config["RATELIMIT_STRATEGY"] = "moving-window"
app.config["RATELIMIT_STORAGE_URL"] = "memory://"

def limiter_key_func():
    # logger.info(f"limiter_key_func called with {session.get('email')}")
    email = None
    if session:
        email = session.get('email')
    if email:
        return email
    # Here, you might want to use a different fallback or even raise an error
    return get_remote_address()
limiter = Limiter(
    app=app,
    key_func=limiter_key_func,
    default_limits=["200 per hour", "10 per minute"]
)
# app.config['PREFERRED_URL_SCHEME'] = 'http' if login_not_needed else 'https'
Session(app)
oauth = OAuth(app)
log = logging.getLogger('werkzeug')
log.setLevel(logging.INFO)
log = logging.getLogger('__main__')
log.setLevel(logging.INFO)
log = logging.getLogger('DocIndex')
log.setLevel(logging.INFO)
log = logging.getLogger('Conversation')
log.setLevel(logging.INFO)
log = logging.getLogger('base')
log.setLevel(logging.INFO)
log = logging.getLogger('faiss.loader')
log.setLevel(logging.INFO)
google = oauth.register(
    name='google',
    client_id=app.config.get("GOOGLE_CLIENT_ID"),
    client_secret=app.config.get("GOOGLE_CLIENT_SECRET"),
    access_token_url='https://accounts.google.com/o/oauth2/token',
    access_token_params=None,
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params=None,
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    userinfo_endpoint='https://openidconnect.googleapis.com/v1/userinfo',  # This is only needed if using openId to fetch user info
    client_kwargs={'scope': 'email profile'},
    server_metadata_url= 'https://accounts.google.com/.well-known/openid-configuration',
)
os.makedirs(os.path.join(os.getcwd(), folder), exist_ok=True)
cache_dir = os.path.join(os.getcwd(), folder, "cache")
users_dir = os.path.join(os.getcwd(), folder, "users")
pdfs_dir = os.path.join(os.getcwd(), folder, "pdfs")
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(users_dir, exist_ok=True)
os.makedirs(pdfs_dir, exist_ok=True)
os.makedirs(os.path.join(folder, "locks"), exist_ok=True)
nlp = English()  # just the language with no model
_ = nlp.add_pipe("lemmatizer")
nlp.initialize()
conversation_folder = os.path.join(os.getcwd(), folder, "conversations")
folder = os.path.join(os.getcwd(), folder, "documents")
os.makedirs(folder, exist_ok=True)
os.makedirs(conversation_folder, exist_ok=True)


cache = Cache(app, config={'CACHE_TYPE': 'filesystem', 'CACHE_DIR': cache_dir, 'CACHE_DEFAULT_TIMEOUT': 7 * 24 * 60 * 60})

def check_login(session):
    email = dict(session).get('email', None)
    name = dict(session).get('name', None)
    logger.debug(f"Check Login for email {session.get('email')} and name {session.get('name')}")
    return email, name, email is not None and name is not None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        logger.debug(f"Login Required call for email {session.get('email')} and name {session.get('name')}")
        if session.get('email') is None or session.get('name') is None:
            return redirect('/login', code=302)
        return f(*args, **kwargs)
    return decorated_function

@app.route('/addUpvoteOrDownvote', methods=['POST'])
@limiter.limit("100 per minute")
@login_required
def add_upvote_downvote():
    email, name, _ = check_login(session)
    data = request.get_json()
    logger.info(f"'/addUpvoteOrDownvote' Get upvote-downvote request with {data}")
    if "question_text" in data:
        source = indexed_docs[data['doc_id']].doc_source if data['doc_id'] in indexed_docs else str(data['doc_id'])
        question_id = str(mmh3.hash(source + data["question_text"], signed=False))
        logger.debug(f"'/addUpvoteOrDownvote' -> generated question_id = {question_id}, Received q_id = {data['question_id']}, both same = {data['question_id'] == question_id}")
        if checkNoneOrEmpty(data['question_id']):
            data['question_id'] = question_id
    if checkNoneOrEmpty(data['question_id']) or checkNoneOrEmpty(data['doc_id']):
        return "Question Id and Doc Id are needed for `/addUpvoteOrDownvote`", 400
    addUpvoteOrDownvote(email, data['question_id'], data['doc_id'], data['upvote'], data['downvote'])
    return jsonify({'status': 'success'}), 200

@app.route('/getUpvotesDownvotesByUser', methods=['GET'])
@limiter.limit("100 per minute")
@login_required
def get_votes_by_user():
    email, name, _ = check_login(session)
    rows = getUpvotesDownvotesByUser(email)
    logger.debug(f"called , response = {rows}")
    return jsonify(rows), 200

# TODO: make bulk api for this
@app.route('/getUpvotesDownvotesByQuestionId/<question_id>', methods=['POST'])
@limiter.limit("5000 per minute")
@login_required
def get_votes_by_question(question_id):
    if checkNoneOrEmpty(question_id) or question_id.strip().lower() == "null":
        data = request.get_json()
        logger.debug(f"'/getUpvotesDownvotesByQuestionId' -> data = {data}")
        if "question_text" in data and "doc_id" in data:
            source = indexed_docs[data['doc_id']].doc_source if data['doc_id'] in indexed_docs else str(data['doc_id'])
            question_id = str(mmh3.hash(source + data["question_text"], signed=False))
        else:
            return "Question Id empty", 400
    email, name, _ = check_login(session)
    rows = getUpvotesDownvotesByQuestionId(question_id)
    logger.info(f"'/getUpvotesDownvotesByQuestionId' called with question_id = {question_id}, response = {rows}")
    return jsonify(rows), 200

# TODO: make bulk api for this
@app.route('/getUpvotesDownvotesByQuestionIdAndUser', methods=['POST'])
@limiter.limit("5000 per minute")
@login_required
def get_votes_by_question_and_user():
    email, name, _ = check_login(session)
    data = request.get_json()
    question_id = data.get('question_id')
    if checkNoneOrEmpty(question_id) or question_id.strip().lower() == "null":

        logger.info(f"'/getUpvotesDownvotesByQuestionIdAndUser' -> data = {data}")
        if "question_text" in data and "doc_id" in data:
            source = indexed_docs[data['doc_id']].doc_source if data['doc_id'] in indexed_docs else str(data['doc_id'])
            question_id = str(mmh3.hash(source + data["question_text"], signed=False))
            logger.debug(f"'/getUpvotesDownvotesByQuestionIdAndUser' -> generated question_id = {question_id}")
        else:
            return "Question Id empty", 400
    rows = getUpvotesDownvotesByQuestionIdAndUser(question_id, email)
    logger.info(f"'/getUpvotesDownvotesByQuestionIdAndUser' called with question_id = {question_id}, response = {rows}")
    return jsonify(rows), 200


@app.route('/addUserQuestionFeedback', methods=['POST'])
@limiter.limit("100 per minute")
@login_required
def add_user_question_feedback():
    email, name, _ = check_login(session)
    data = request.get_json()
    logger.info(f"Get granular feedback request with {data}")
    if "question_text" in data:
        source = indexed_docs[data['doc_id']].doc_source if data['doc_id'] in indexed_docs else str(data['doc_id'])
        question_id = str(mmh3.hash(source + data["question_text"], signed=False))
        logger.debug(f"'/addUserQuestionFeedback' -> generated question_id = {question_id}, Received q_id = {data['question_id']}, both same = {data['question_id'] == question_id}")
        if checkNoneOrEmpty(data['question_id']):
            data['question_id'] = question_id
    if checkNoneOrEmpty(data['question_id']) or checkNoneOrEmpty(data['doc_id']):
        return "Question Id and Doc Id are needed for `/addUserQuestionFeedback`", 400
    try:
        addGranularFeedback(email, data['question_id'], data['feedback_type'], data['feedback_items'], data['comments'], data['question_text'])
        return jsonify({'status': 'success'}), 200
    except ValueError as e:
        return str(e), 400


@app.route('/write_review/<doc_id>/<tone>', methods=['GET'])
@limiter.limit("5 per minute")
@login_required
def write_review(doc_id, tone):
    keys = keyParser(session)
    email, name, _ = check_login(session)
    if tone == 'undefined':
        tone = "none"
    assert tone in ["positive", "negative", "neutral", "none"]
    review_topic = request.args.get('review_topic') # Has top level key and then an index variable
    review_topic = review_topic.split(",")
    review_topic = [r for r in review_topic if len(r.strip()) > 0 and r.strip()!='null']
    if len(review_topic) > 1:
        review_topic = [str(review_topic[0]), int(review_topic[1])]
    else:
        review_topic = review_topic[0]
    additional_instructions = request.args.get('instruction')
    use_previous_reviews = int(request.args.get('use_previous_reviews'))
    score_this_review = int(request.args.get('score_this_review'))
    is_meta_review = int(request.args.get('is_meta_review'))
    
    review = set_keys_on_docs(indexed_docs[doc_id], keys).get_review(tone, review_topic, additional_instructions, score_this_review, use_previous_reviews, is_meta_review)
    return Response(stream_with_context(review), content_type='text/plain')

@app.route('/get_reviews/<doc_id>', methods=['GET'])
@limiter.limit("10 per minute")
@login_required
def get_all_reviews(doc_id):
    keys = keyParser(session)
    email, name, _ = check_login(session)
    reviews = set_keys_on_docs(indexed_docs[doc_id], keys).get_all_reviews()
    # lets send json response
    return jsonify(reviews)
    
    
@app.route('/login')
@limiter.limit("5 per minute")
def login():
    scheme_is_https = request.headers.get('X-Forwarded-Proto', 'http') == "https"
    redirect_uri = url_for('authorize', _external=True)
    redirect_uri = convert_http_to_https(redirect_uri) if scheme_is_https else redirect_uri
    if login_not_needed:
        logger.info(f"Login not needed send login.html")
        email = request.args.get('email')
        if email is None:
            return send_from_directory('interface', 'login.html', max_age=0)
        session['email'] = email
        session['name'] = email
        addUserToDoc(email, "3408472793", "https://arxiv.org/pdf/1706.03762.pdf")
        return redirect('/interface', code=302)
    else:
        logger.info(f"Login needed with redirect authorize uri = {redirect_uri}")
        return google.authorize_redirect(redirect_uri)

@app.route('/logout')
@limiter.limit("1 per minute")
@login_required
def logout():
    
    if 'token' in session: 
        access_token = session['token']['access_token'] 
        requests.post('https://accounts.google.com/o/oauth2/revoke',
            params={'token': access_token},
            headers = {'content-type': 'application/x-www-form-urlencoded'})
    
    session.clear() # clears the session
    return render_template_string("""
        <h1>Logged out</h1>
        <p><a href="{{ url_for('login') }}">Click here</a> to log in again. You can now close this Tab/Window.</p>
    """)

@app.route('/authorize')
@limiter.limit("15 per minute")
def authorize():
    logger.info(f"Authorize for email {session.get('email')} and name {session.get('name')}")
    token = google.authorize_access_token()
    resp = google.get('userinfo')
    if resp.ok:
        user_info = resp.json()
        session['email'] = user_info['email']
        session['name'] = user_info['name']
        session['token'] = token
        return redirect('/interface')
    else:
        return "Failed to log in", 401

@app.route('/get_user_info')
@limiter.limit("15 per minute")
@login_required
def get_user_info():
    if 'email' in session and "name" in session:
        return jsonify(name=session['name'], email=session['email'])
    elif google.authorized:
        resp = google.get('userinfo')
        if resp.ok:
            session['email'] = resp.json()['email']
            session['name'] = resp.json()['name']
            return jsonify(name=resp.json()["name"], email=resp.json()["email"])
    else:
        return "Not logged in", 401

class IndexDict(dict):
    def __getitem__(self, key):
        try:
            item = super().__getitem__(key)
            if item.doc_id in doc_index_cache:
                return item
            else:
                item = item.copy()
            doc_index_cache.add(item.doc_id)
            super().__setitem__(key, item)
            return item
        except KeyError:
            exc = traceback.format_exc()
            logger.error(f"Error in getting doc_index for key = {key}, error = {exc}")
            return load_document(folder, key)
    
    def __setitem__(self, __key: str, __value: DocIndex) -> None:
        __value = __value.copy()
        return super().__setitem__(__key, __value)
indexed_docs: IndexDict[str, DocIndex] = IndexDict()
doc_index_cache = SetQueue(maxsize=50)
def load_conversation(conversation_id):
    return Conversation.load_local(os.path.join(conversation_folder, conversation_id))

conversation_cache = DefaultDictQueue(maxsize=50, default_factory=load_conversation)
    
def set_keys_on_docs(docs, keys):
    logger.debug(f"Attaching keys to doc")
    if isinstance(docs, dict):
        # docs = {k: v.copy() for k, v in docs.items()}
        for k, v in docs.items():
            v.set_api_keys(keys)
    elif isinstance(docs, (list, tuple, set)):
        # docs = [d.copy() for d in docs]
        for d in docs:
            d.set_api_keys(keys)
    else:
        assert isinstance(docs, (DocIndex, ImmediateDocIndex, Conversation))
        docs.set_api_keys(keys)
    return docs
    

# Initialize an empty list of documents for BM25
bm25_corpus: List[List[str]] = []
doc_id_to_bm25_index: Dict[str, int] = {}
bm25 = [None]
    
def get_bm25_grams(text):
    unigrams = text.split()
    bigrams = generate_ngrams(unigrams, 2)
    trigrams = generate_ngrams(unigrams, 3)
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    bigrams_lemma = generate_ngrams(lemmas, 2)
    trigrams_lemma = generate_ngrams(lemmas, 3)
    return unigrams + bigrams + trigrams + lemmas + bigrams_lemma + trigrams_lemma

def add_to_bm25_corpus(doc_index: DocIndex):
    global bm25_corpus, doc_id_to_bm25_index
    try:
        doc_info = doc_index.get_short_info()
        text = doc_info['title'].lower() + " " + doc_info['short_summary'].lower() + " " + doc_info['summary'].lower()
    except Exception as e:
        logger.warning(f"Error in getting text for doc_id = {doc_index.doc_id}, error = {e}")
        text = doc_index.indices["chunks"][0].lower()
    bm25_corpus.append(get_bm25_grams(text))
    doc_id_to_bm25_index[doc_index.doc_id] = len(bm25_corpus) - 1
    bm25[0] = BM25Okapi(bm25_corpus)

def load_documents(folder):
    global indexed_docs, bm25_corpus, doc_id_to_bm25_index
    folders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
    docs: List[DocIndex] = [DocIndex.load_local(os.path.join(folder, filepath)) for filepath in folders]
    docs = [doc for doc in docs if doc is not None] # and doc.visible
    # filename = os.path.basename(filepath)
    for doc_index in docs:
        indexed_docs[doc_index.doc_id] = doc_index
        add_to_bm25_corpus(doc_index)

def load_document(folder, filepath):
    global indexed_docs, bm25_corpus, doc_id_to_bm25_index
    doc_index = DocIndex.load_local(os.path.join(folder, filepath))
    indexed_docs[doc_index.doc_id] = doc_index
    add_to_bm25_corpus(doc_index)
    return doc_index



@app.route('/search_document', methods=['GET'])
@limiter.limit("1000 per minute")
@login_required
def search_document():
    keys = keyParser(session)
    email, name, loggedin = check_login(session)
    docs = getDocsForUser(email)
    doc_ids = [d[1] for d in docs]
    
    search_text = request.args.get('text')
    if search_text:
        search_text = search_text.strip().lower()
        bm = bm25[0]
        search_tokens = get_bm25_grams(search_text)
        scores = bm.get_scores(search_tokens)
        results = sorted([(score, doc_id) for doc_id, score in zip(indexed_docs.keys(), scores)], reverse=True)
        docs = [set_keys_on_docs(indexed_docs[doc_id], keys) for score, doc_id in results[:4] if doc_id in doc_ids]
        scores = [score for score, doc_id in results[:4] if doc_id in doc_ids]
        top_results = [doc.get_short_info() for score, doc in zip(scores, docs)]
        logger.debug(f"Search results = {[(score, doc.doc_source) for score, doc in zip(scores, docs)]}")
        return jsonify(top_results)
    else:
        return jsonify({'error': 'No search text provided'}), 400


@app.route('/list_all', methods=['GET'])
@limiter.limit("100 per minute")
@login_required
def list_all():
    keys = keyParser(session)
    email, name, loggedin = check_login(session)
    docs = getDocsForUser(email)
    if len(docs) == 0:
        addUserToDoc(email, "3408472793", "https://arxiv.org/pdf/1706.03762.pdf")
        docs = getDocsForUser(email)
    doc_ids = set([d[1] for d in docs])
    docs = [set_keys_on_docs(indexed_docs[docId], keys).get_short_info() for docId in doc_ids if docId in indexed_docs]
    # docs = sorted(docs, key=lambda x: x['last_updated'], reverse=True)
    return jsonify(docs)


@app.route('/get_document_detail', methods=['GET'])
@limiter.limit("100 per minute")
@login_required
def get_document_detail():
    keys = keyParser(session)
    doc_id = request.args.get('doc_id')
    logger.info(f"/get_document_detail for doc_id = {doc_id}, doc present = {doc_id in indexed_docs}")
    if doc_id in indexed_docs:
        
        return jsonify(set_keys_on_docs(indexed_docs[doc_id], keys).get_all_details())
    else:
        return jsonify({'error': 'Document not found'}), 404
    


@app.route('/index_document', methods=['POST'])
@limiter.limit("10 per minute")
@login_required
def index_document():
    keys = keyParser(session)
    email, name, loggedin = check_login(session)
    
    pdf_url = request.json.get('pdf_url')
    # if "arxiv.org" not in pdf_url:
    #     return jsonify({'error': 'Only arxiv urls are supported at this moment.'}), 400
    pdf_url = convert_to_pdf_link_if_needed(pdf_url)
    if pdf_url:
        try:
            doc_index = immediate_create_and_save_index(pdf_url, keys)
            addUserToDoc(email, doc_index.doc_id, doc_index.doc_source)
            return jsonify({'status': 'Indexing started', 'doc_id': doc_index.doc_id, "properly_indexed": doc_index.doc_id in indexed_docs})
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'No pdf_url provided'}), 400

@app.route('/set_keys', methods=['POST'])
@limiter.limit("100 per minute")
@login_required
def set_keys():
    keys = request.json  # Assuming keys are sent as JSON in the request body
    for key, value in keys.items():
        session[key] = value
    return jsonify({'result': 'success'})

@app.route('/clear_session', methods=['GET'])
@limiter.limit("100 per minute")
@login_required
def clear_session():
    # clear the session
    session.clear()
    return jsonify({'result': 'session cleared'})


def delayed_execution(func, delay, *args):
    time.sleep(delay)
    return func(*args)

    
def immediate_create_and_save_index(pdf_url, keys):
    pdf_url = pdf_url.strip()
    matching_docs = [v for k, v in indexed_docs.items() if v.doc_source==pdf_url]
    if len(matching_docs) == 0:
        doc_index = create_immediate_document_index(pdf_url, folder, keys)
        doc_index = set_keys_on_docs(doc_index, keys)
        save_index(doc_index, folder)
    else:
        logger.info(f"{pdf_url} is already indexed")
        doc_index = matching_docs[0]
    return doc_index
    
def save_index(doc_index: DocIndex, folder):
    indexed_docs[doc_index.doc_id] = doc_index
    add_to_bm25_corpus(doc_index)
    doc_index.save_local()

    
@app.route('/streaming_get_answer', methods=['POST'])
@limiter.limit("15 per minute")
@login_required
def streaming_get_answer():
    keys = keyParser(session)
    additional_docs_to_read = request.json.get("additional_docs_to_read", [])
    a = use_multiple_docs = request.json.get("use_multiple_docs", False) and isinstance(additional_docs_to_read, (tuple, list)) and len(additional_docs_to_read) > 0
    b = use_references_and_citations = request.json.get("use_references_and_citations", False)
    c = provide_detailed_answers = request.json.get("provide_detailed_answers", False)
    d = perform_web_search = request.json.get("perform_web_search", False)
    if not (sum([a, b, c, d]) == 0 or sum([a, b, c, d]) == 1):
        return Response("Invalid answering strategy passed.", status=400,  content_type='text/plain')
    provide_detailed_answers = int(provide_detailed_answers)
    if use_multiple_docs:
        additional_docs_to_read = [set_keys_on_docs(indexed_docs[doc_id], keys) for doc_id in additional_docs_to_read]
    meta_fn = defaultdict(lambda: False, dict(additional_docs_to_read=additional_docs_to_read, use_multiple_docs=use_multiple_docs, use_references_and_citations=use_references_and_citations, provide_detailed_answers=provide_detailed_answers, perform_web_search=perform_web_search))
    
    doc_id = request.json.get('doc_id')
    query = request.json.get('query')
    if doc_id in indexed_docs:
        answer = set_keys_on_docs(indexed_docs[doc_id], keys).streaming_get_short_answer(query, meta_fn)
        return Response(stream_with_context(answer), content_type='text/plain')
    else:
        return Response("Error Document not found", status=404,  content_type='text/plain')
    
@app.route('/streaming_summary', methods=['GET'])
@limiter.limit("5 per minute")
@login_required
def streaming_summary():
    keys = keyParser(session)
    doc_id = request.args.get('doc_id')
    if doc_id in indexed_docs:
        doc = set_keys_on_docs(indexed_docs[doc_id], keys)
        answer = doc.streaming_build_summary()
        p = multiprocessing.Process(target=delayed_execution, args=(save_index, 180, doc, folder))
        p.start()
        return Response(stream_with_context(answer), content_type='text/plain')
    else:
        return Response("Error Document not found", content_type='text/plain')
    
    

@app.route('/streaming_get_followup_answer', methods=['POST'])
@limiter.limit("5 per minute")
@login_required
def streaming_get_followup_answer():
    keys = keyParser(session)
    additional_docs_to_read = request.json.get("additional_docs_to_read", [])
    a = use_multiple_docs = request.json.get("use_multiple_docs", False) and isinstance(additional_docs_to_read, (tuple, list)) and len(additional_docs_to_read) > 0
    b = use_references_and_citations = request.json.get("use_references_and_citations", False)
    c = provide_detailed_answers = request.json.get("provide_detailed_answers", False)
    d = perform_web_search = request.json.get("perform_web_search", False)
    if not (sum([a, b, c, d]) == 0 or sum([a, b, c, d]) == 1):
        return Response("Invalid answering strategy passed.", status=400,  content_type='text/plain')
    provide_detailed_answers = int(provide_detailed_answers)
    if use_multiple_docs:
        additional_docs_to_read = [set_keys_on_docs(indexed_docs[doc_id], keys) for doc_id in additional_docs_to_read]
    meta_fn = defaultdict(lambda: False, dict(additional_docs_to_read=additional_docs_to_read, use_multiple_docs=use_multiple_docs, use_references_and_citations=use_references_and_citations, provide_detailed_answers=provide_detailed_answers, perform_web_search=perform_web_search))
    
    doc_id = request.json.get('doc_id')
    query = request.json.get('query')
    previous_answer = request.json.get('previous_answer')
    if doc_id in indexed_docs:
        answer = set_keys_on_docs(indexed_docs[doc_id], keys).streaming_ask_follow_up(query, previous_answer, meta_fn)
        return Response(stream_with_context(answer), content_type='text/plain')
    else:
        return Response("Error Document not found", content_type='text/plain')

from multiprocessing import Lock

lock = Lock()

@app.route('/delete_document', methods=['DELETE'])
@limiter.limit("30 per minute")
@login_required
def delete_document():
    email, name, loggedin = check_login(session)
    doc_id = request.args.get('doc_id')
    if not doc_id or doc_id not in indexed_docs:
        return jsonify({'error': 'Document not found'}), 404
    removeUserFromDoc(email, doc_id)

    return jsonify({'status': 'Document deleted successfully'}), 200

@app.route('/get_paper_details', methods=['GET'])
@limiter.limit("5 per minute")
@login_required
def get_paper_details():
    keys = keyParser(session)
    doc_id = request.args.get('doc_id')
    if doc_id in indexed_docs:
        paper_details = set_keys_on_docs(indexed_docs[doc_id], keys).paper_details
        return jsonify(paper_details)
    else:
        return jsonify({'error': 'Document not found'}), 404

@app.route('/refetch_paper_details', methods=['GET'])
@limiter.limit("5 per minute")
@login_required
def refetch_paper_details():
    keys = keyParser(session)
    doc_id = request.args.get('doc_id')
    if doc_id in indexed_docs:
        paper_details = set_keys_on_docs(indexed_docs[doc_id], keys).refetch_paper_details()
        return jsonify(paper_details)
    else:
        return jsonify({'error': 'Document not found'}), 404

@app.route('/get_extended_abstract', methods=['GET'])
@limiter.limit("5 per minute")
@login_required
def get_extended_abstract():
    keys = keyParser(session)
    doc_id = request.args.get('doc_id')
    paper_id = request.args.get('paper_id')
    if doc_id in indexed_docs:
        extended_abstract = set_keys_on_docs(indexed_docs[doc_id], keys).get_extended_abstract_for_ref_or_cite(paper_id)
        return Response(stream_with_context(extended_abstract), content_type='text/plain')
    else:
        return Response("Error Document not found", content_type='text/plain')
    
@app.route('/get_fixed_details', methods=['GET'])
@limiter.limit("30 per minute")
@login_required
def get_fixed_details():
    keys = keyParser(session)
    doc_id = request.args.get('doc_id')
    detail_key = request.args.get('detail_key')
    if doc_id in indexed_docs:
        fixed_details = set_keys_on_docs(indexed_docs[doc_id], keys).get_fixed_details(detail_key)
        return Response(stream_with_context(fixed_details), content_type='text/plain')
    else:
        return Response("Error: Document not found", content_type='text/plain')


    
from flask import send_from_directory

@app.route('/favicon.ico')
@limiter.limit("30 per minute")
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')
    
@app.route('/loader.gif')
@limiter.limit("10 per minute")
def loader():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'gradient-loader.gif', mimetype='image/gif')

@app.route('/interface/<path:path>')
@limiter.limit("100 per minute")
def send_static(path):
    return send_from_directory('interface', path, max_age=0)

@app.route('/interface')
@limiter.limit("20 per minute")
@login_required
def interface():
    return send_from_directory('interface', 'interface.html', max_age=0)

from flask import Response, stream_with_context

@app.route('/proxy', methods=['GET'])
@login_required
def proxy():
    file_url = request.args.get('file')
    logger.debug(f"Proxying file {file_url}, exists on disk = {os.path.exists(file_url)}")
    return Response(stream_with_context(cached_get_file(file_url)), mimetype='application/pdf')

@app.route('/')
@limiter.limit("20 per minute")
@login_required
def index():
    return redirect('/interface')

@app.route('/upload_pdf', methods=['POST'])
@limiter.limit("10 per minute")
@login_required
def upload_pdf():
    keys = keyParser(session)
    email, name, loggedin = check_login(session)
    pdf_file = request.files.get('pdf_file')
    if pdf_file:
        try:
            # Determine the file extension
            file_ext = os.path.splitext(pdf_file.filename)[1]

            # Save the original file first
            original_file_path = os.path.join(pdfs_dir, pdf_file.filename)
            pdf_file.save(original_file_path)

            if file_ext in ['.doc', '.docx']:
                # Convert to PDF
                pdf_filename = os.path.splitext(pdf_file.filename)[0] + ".pdf"
                pdf_file_path = os.path.join(pdfs_dir, pdf_filename)

                if not convert_doc_to_pdf(original_file_path, pdf_file_path):
                    return jsonify({'error': 'Conversion to PDF failed'}), 400
            else:
                pdf_file_path = original_file_path

            # Create and save index
            doc_index = immediate_create_and_save_index(pdf_file_path, keys)

            addUserToDoc(email, doc_index.doc_id, doc_index.doc_source)
            return jsonify({'status': 'Indexing started', 'doc_id': doc_index.doc_id,
                            "properly_indexed": doc_index.doc_id in indexed_docs})

        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'No pdf_file provided'}), 400


@app.route('/upload_doc_to_conversation/<conversation_id>', methods=['POST'])
@limiter.limit("10 per minute")
@login_required
def upload_doc_to_conversation(conversation_id):
    keys = keyParser(session)
    email, name, loggedin = check_login(session)
    pdf_file = request.files.get('pdf_file')
    conversation: Conversation = conversation_cache[conversation_id]
    conversation = set_keys_on_docs(conversation, keys)
    if pdf_file and conversation_id:
        try:
            # save file to disk at pdfs_dir.
            pdf_file.save(os.path.join(pdfs_dir, pdf_file.filename))
            full_pdf_path = os.path.join(pdfs_dir, pdf_file.filename)
            conversation.add_uploaded_document(full_pdf_path)
            conversation.save_local()
            return jsonify({'status': 'Indexing started'})
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 400
    # If it is not a pdf file then assume we have url
    conversation: Conversation = conversation_cache[conversation_id]
    conversation = set_keys_on_docs(conversation, keys)
    pdf_url = request.json.get('pdf_url')
    pdf_url = convert_to_pdf_link_if_needed(pdf_url)
    if pdf_url:
        try:
            conversation.add_uploaded_document(pdf_url)
            conversation.save_local()
            return jsonify({'status': 'Indexing started'})
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'No pdf_url or pdf_file provided'}), 400

@app.route('/delete_document_from_conversation/<conversation_id>/<document_id>', methods=['DELETE'])
@limiter.limit("10 per minute")
@login_required
def delete_document_from_conversation(conversation_id, document_id):
    keys = keyParser(session)
    email, name, loggedin = check_login(session)
    conversation: Conversation = conversation_cache[conversation_id]
    conversation = set_keys_on_docs(conversation, keys)
    doc_id = document_id
    if doc_id:
        try:
            conversation.delete_uploaded_document(doc_id)
            return jsonify({'status': 'Document deleted'})
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'No doc_id provided'}), 400

@app.route('/list_documents_by_conversation/<conversation_id>', methods=['GET'])
@limiter.limit("30 per minute")
@login_required
def list_documents_by_conversation(conversation_id):
    keys = keyParser(session)
    email, name, loggedin = check_login(session)
    conversation: Conversation = conversation_cache[conversation_id]
    conversation = set_keys_on_docs(conversation, keys)
    if conversation:
        docs:List[DocIndex] = conversation.get_uploaded_documents(readonly=True)
        docs = [d.get_short_info() for d in docs]
        # sort by doc_id
        # docs = sorted(docs, key=lambda x: x['doc_id'], reverse=True)
        return jsonify(docs)
    else:
        return jsonify({'error': 'Conversation not found'}), 404

@app.route('/download_doc_from_conversation/<conversation_id>/<doc_id>', methods=['GET'])
@limiter.limit("30 per minute")
@login_required
def download_doc_from_conversation(conversation_id, doc_id):
    keys = keyParser(session)
    conversation: Conversation = conversation_cache[conversation_id]
    conversation = set_keys_on_docs(conversation, keys)
    if conversation:
        doc:DocIndex = conversation.get_uploaded_documents(doc_id, readonly=True)[0]
        if doc and os.path.exists(doc.doc_source):
            return send_from_directory(doc.doc_source, as_attachment=True)
        elif doc:
            return redirect(doc.doc_source)
        else:
            return jsonify({'error': 'Document not found'}), 404
    else:
        return jsonify({'error': 'Conversation not found'}), 404

def cached_get_file(file_url):
    chunk_size = 1024  # Define your chunk size
    file_data = cache.get(file_url)

    # If the file is not in the cache, read it from disk and save it to the cache
    if file_data is not None:
        logger.info(f"cached_get_file for {file_url} found in cache")
        for chunk in file_data:
            yield chunk
        # how do I chunk with chunk size?

    elif os.path.exists(file_url):
        file_data = []
        with open(file_url, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if chunk:
                    file_data.append(chunk)
                    yield chunk
                if not chunk:
                    break
        cache.set(file_url, file_data)
    else:   
        file_data = []
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
            req = requests.get(file_url, stream=True,
                               verify=False, headers=headers)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download file: {e}")
            req = requests.get(file_url, stream=True, verify=False)
        # TODO: save the downloaded file to disk.
        
        for chunk in req.iter_content(chunk_size=chunk_size):
            file_data.append(chunk)
            yield chunk
        cache.set(file_url, file_data)


### chat apis
@app.route('/list_conversation_by_user', methods=['GET'])
@limiter.limit("50 per minute")
@login_required
def list_conversation_by_user():
    # TODO: sort by last_updated
    email, name, loggedin = check_login(session)
    keys = keyParser(session)
    last_n_conversations = request.args.get('last_n_conversations', 10)
    # TODO: add ability to get only n conversations
    conversation_ids = [c[1] for c in getCoversationsForUser(email)]
    conversations = [conversation_cache[conversation_id] for conversation_id in conversation_ids]
    conversations = [conversation for conversation in conversations if conversation is not None]
    conversations = [set_keys_on_docs(conversation, keys) for conversation in conversations]
    data = [[conversation.get_metadata(), conversation] for conversation in conversations]
    sorted_data_reverse = sorted(data, key=lambda x: x[0]['last_updated'], reverse=True)
    # TODO: if any conversation has 0 messages then just make it the latest. IT should also have a zero length summary.

    if len(sorted_data_reverse) > 0 and len(sorted_data_reverse[0][0]["summary_till_now"].strip()) > 0:
        sorted_data_reverse = sorted(sorted_data_reverse, key=lambda x: len(x[0]['summary_till_now'].strip()), reverse=False)
        if sorted_data_reverse[0][0]["summary_till_now"].strip() == "":
            new_conversation = sorted_data_reverse[0][1]
            sorted_data_reverse = sorted_data_reverse[1:]
            sorted_data_reverse = sorted(sorted_data_reverse, key=lambda x: x[0]['last_updated'], reverse=True)
            new_conversation.set_field("memory", {"last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        else:
            new_conversation = create_conversation_simple(session)
        sorted_data_reverse.insert(0, [new_conversation.get_metadata(), new_conversation])
    sorted_data_reverse = [sd[0] for sd in sorted_data_reverse]
    return jsonify(sorted_data_reverse)

@app.route('/create_conversation', methods=['POST'])
@limiter.limit("5 per minute")
@login_required
def create_conversation():
    conversation = create_conversation_simple(session)
    data = conversation.get_metadata()
    return jsonify(data)

def create_conversation_simple(session):
    email, name, loggedin = check_login(session)
    keys = keyParser(session)
    from base import get_embedding_model
    conversation_id = email + "_" + ''.join(secrets.choice(alphabet) for i in range(36))
    conversation = Conversation(email, openai_embed=get_embedding_model(keys), storage=conversation_folder,
                                conversation_id=conversation_id)
    conversation = set_keys_on_docs(conversation, keys)
    addConversationToUser(email, conversation.conversation_id)
    return conversation


@app.route('/list_messages_by_conversation/<conversation_id>', methods=['GET'])
@limiter.limit("100 per minute")
@login_required
def list_messages_by_conversation(conversation_id):
    keys = keyParser(session)
    email, name, loggedin = check_login(session)
    last_n_messages = request.args.get('last_n_messages', 10)
    # TODO: add capability to get only last n messages
    conversation_ids = [c[1] for c in getCoversationsForUser(email)]
    if conversation_id not in conversation_ids:
        return jsonify({"message": "Conversation not found"}), 404
    else:
        conversation = conversation_cache[conversation_id]
        conversation = set_keys_on_docs(conversation, keys)
    return jsonify(conversation.get_message_list())

@app.route('/list_messages_by_conversation_shareable/<conversation_id>', methods=['GET'])
@limiter.limit("10 per minute")
def list_messages_by_conversation_shareable(conversation_id):
    keys = keyParser(session)
    email, name, loggedin = check_login(session)
    conversation_ids = [c[1] for c in getAllCoversations()]
    if conversation_id not in conversation_ids:
        return jsonify({"message": "Conversation not found"}), 404
    else:
        conversation: Conversation = conversation_cache[conversation_id]

    if conversation:
        docs: List[DocIndex] = conversation.get_uploaded_documents(readonly=True)
        docs = [d.get_short_info() for d in docs]
        messages = conversation.get_message_list()
        return jsonify({"messages": messages, "docs": docs})
    else:
        return jsonify({'error': 'Conversation not found'}), 404

@app.route('/send_message/<conversation_id>', methods=['POST'])
@limiter.limit("5 per minute")
@login_required
def send_message(conversation_id):
    keys = keyParser(session)
    email, name, loggedin = check_login(session)
    conversation_ids = [c[1] for c in getCoversationsForUser(email)]
    if conversation_id not in conversation_ids:
        return jsonify({"message": "Conversation not found"}), 404
    else:
        conversation = conversation_cache[conversation_id]
        conversation = set_keys_on_docs(conversation, keys)

    query = request.json
    query["additional_docs_to_read"] = []
    additional_params: dict = query["checkboxes"]
    additional_docs_to_read = additional_params.get("additional_docs_to_read", [])
    additional_docs_to_read = list(set(additional_docs_to_read))
    use_multiple_docs = additional_params.get("use_multiple_docs", False) and isinstance(additional_docs_to_read,
                                                                                         (tuple, list)) and len(
        additional_docs_to_read) > 0
    if use_multiple_docs:
        keys = copy.deepcopy(keys)
        keys["use_gpt4"] = False
        additional_docs_to_read = [set_keys_on_docs(indexed_docs[doc_id], keys) for doc_id in
                                   additional_docs_to_read]
        query["additional_docs_to_read"] = additional_docs_to_read

    # We don't process the request data in this mockup, but we would normally send a new message here
    return Response(stream_with_context(conversation(query)), content_type='text/plain')


@app.route('/get_conversation_details/<conversation_id>', methods=['GET'])
@limiter.limit("100 per minute")
@login_required
def get_conversation_details(conversation_id):
    keys = keyParser(session)
    email, name, loggedin = check_login(session)

    conversation_ids = [c[1] for c in getCoversationsForUser(email)]
    if conversation_id not in conversation_ids:
        return jsonify({"message": "Conversation not found"}), 404
    else:
        conversation = conversation_cache[conversation_id]
        conversation = set_keys_on_docs(conversation, keys)
    # Dummy data
    data = conversation.get_metadata()
    return jsonify(data)

@app.route('/delete_conversation/<conversation_id>', methods=['DELETE'])
@limiter.limit("25 per minute")
@login_required
def delete_conversation(conversation_id):
    email, name, loggedin = check_login(session)
    keys = keyParser(session)
    conversation_ids = [c[1] for c in getCoversationsForUser(email)]
    if conversation_id not in conversation_ids:
        return jsonify({"message": "Conversation not found"}), 404
    else:
        conversation = conversation_cache[conversation_id]
        conversation = set_keys_on_docs(conversation, keys)
    removeUserFromConversation(email, conversation_id)
    # In a real application, you'd delete the conversation here
    return jsonify({'message': f'Conversation {conversation_id} deleted'})
@app.route('/delete_message_from_conversation/<conversation_id>/<message_id>/<index>', methods=['DELETE'])
@limiter.limit("30 per minute")
@login_required
def delete_message_from_conversation(conversation_id, message_id, index):
    email, name, loggedin = check_login(session)
    keys = keyParser(session)
    conversation_ids = [c[1] for c in getCoversationsForUser(email)]
    if conversation_id not in conversation_ids:
        return jsonify({"message": "Conversation not found"}), 404
    else:
        conversation = conversation_cache[conversation_id]
        conversation = set_keys_on_docs(conversation, keys)
    conversation.delete_message(message_id, index)
    # In a real application, you'd delete the conversation here
    return jsonify({'message': f'Message {message_id} deleted'})

@app.route('/delete_last_message/<conversation_id>', methods=['DELETE'])
@limiter.limit("30 per minute")
@login_required
def delete_last_message(conversation_id):
    message_id=1
    email, name, loggedin = check_login(session)
    keys = keyParser(session)
    conversation_ids = [c[1] for c in getCoversationsForUser(email)]
    if conversation_id not in conversation_ids:
        return jsonify({"message": "Conversation not found"}), 404
    else:
        conversation = conversation_cache[conversation_id]
        conversation: Conversation = set_keys_on_docs(conversation, keys)
    conversation.delete_last_turn()
    # In a real application, you'd delete the conversation here
    return jsonify({'message': f'Message {message_id} deleted'})

def open_browser(url):
    import webbrowser
    import subprocess
    if sys.platform.startswith('linux'):
        subprocess.call(['xdg-open', url])
    elif sys.platform.startswith('darwin'):
        subprocess.call(['open', url])
    else:
        webbrowser.open(url)
    
create_tables()
load_documents(folder)

# def removeAllUsersFromConversation():
#     conn = create_connection("{}/users.db".format(users_dir))
#     cur = conn.cursor()
#     cur.execute("DELETE FROM UserToConversationId")
#     conn.commit()
#     conn.close()
#
# removeAllUsersFromConversation()

if __name__ == '__main__':
    
    port = 443
   # app.run(host="0.0.0.0", port=port,threaded=True, ssl_context=('cert-ext.pem', 'key-ext.pem'))
    app.run(host="0.0.0.0", port=5000,threaded=True)

