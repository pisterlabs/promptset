from flask import Flask, request, jsonify, make_response, render_template, redirect, url_for, session
from flask import Markup
from collections import defaultdict
import operator
import numpy as np
import json
import time 
import math

import tiktoken
import os
import openai
from tqdm import tqdm
import ast
import re
import uuid, hashlib, datetime, math, urllib


from flask_mysqldb import MySQL
# from flask_cors import CORS


from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func
from sqlalchemy.orm import sessionmaker




application = app = Flask(__name__)
# cors = CORS(application, resources={r"/api/*": {"origins": ["http://127.0.0.1:8887","http://localhost:5000"]}})

# connect to DB
MYSQL_HOST = os.environ['RHETANN_MYSQL_HOST']
MYSQL_USER = os.environ['RHETANN_MYSQL_USER']
MYSQL_PASSWORD = os.environ['RHETANN_MYSQL_PASSWORD']
MYSQL_DB = os.environ['RHETANN_MYSQL_DB']

application.config['MYSQL_HOST'] = MYSQL_HOST
application.config['MYSQL_USER'] = MYSQL_USER
application.config['MYSQL_PASSWORD'] = MYSQL_PASSWORD
application.config['MYSQL_DB'] = MYSQL_DB




application.config['SQLALCHEMY_DATABASE_URI'] =\
        'mysql://'+MYSQL_USER+':'+MYSQL_PASSWORD+'@'+MYSQL_HOST+'/'+MYSQL_DB
application.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

from sqlalchemy import create_engine
engine = create_engine(application.config['SQLALCHEMY_DATABASE_URI'], echo = True)

Session = sessionmaker(bind = engine)
sa_session = Session()

db = SQLAlchemy(application)



class Annotations(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    json_string     = db.Column(db.Text())
    annotator_id    = db.Column(db.Integer)
    sentence_id     = db.Column(db.Integer)
    review_flag     = db.Column(db.Integer)
    completed       = db.Column(db.Integer)
    date_updated    = db.Column(db.DateTime(timezone=True),
                           server_default=func.now())

    def __repr__(self):
        return f'<annotation_id {self.id}>'

class AllSentences(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    article_id      = db.Column(db.String(120))
    technique       = db.Column(db.String(120))
    offsets         = db.Column(db.String(120))
    label           = db.Column(db.Integer)
    text            = db.Column(db.Text())
    parseTree       = db.Column(db.Text())
    date_updated    = db.Column(db.DateTime(timezone=True),
                           server_default=func.now())
    completed       = db.Column(db.Integer)


    def __repr__(self):
        return f'<sentence_id {self.id}>'

class SampleSentences(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    article_id      = db.Column(db.String(120))
    technique       = db.Column(db.String(120))
    offsets         = db.Column(db.String(120))
    label           = db.Column(db.Integer)
    text            = db.Column(db.Text())
    parseTree       = db.Column(db.Text())
    date_updated    = db.Column(db.DateTime(timezone=True),
                           server_default=func.now())
    completed       = db.Column(db.Integer)


    def __repr__(self):
        return f'<sentence_id {self.id}>'


application.static_folder = 'ui'

application.secret_key = 'yoursecretkey'
 
mysql = MySQL(application)




# http://localhost:5000/ - the following will be our login page, which will use both GET and POST requests
@application.route('/', methods=['GET', 'POST'])
def login():

    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('home.html', userid=session['id'],username=session['username'], firstname=session['firstname'])



    # Output message if something goes wrong...
    msg = ''
    username = ''
    password = ''

    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        # Create variables for easy access
        
        password = request.form['password']
        email = request.form['email']
        username = email

        
        # Retrieve the hashed password
        hash = password + application.secret_key
        hash = hashlib.sha1(hash.encode())
        password = hash.hexdigest();
        

        # Check if account exists using MySQL
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM annotators WHERE username = %s AND password = %s', (username, password,))
        # Fetch one record and return result
        account = cursor.fetchone()

        
        # account (4, None, None, 'test@test.com', 'test', 'test') --> (id, firstname, lastname, email, username, password)

        # If account exists in accounts table in our database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account[0]
            session['username'] = account[4]
            session['firstname'] = account[1]
            # Redirect to home 
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'


    # Show the login form with message (if any)
    return render_template('index.html', msg=msg)


# http://localhost:5000/logout - this will be the logout page
@application.route('/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   session.pop('firstname', None)
   # Redirect to login page
   return redirect(url_for('login'))


# http://localhost:5000/register - this will be the registration page, we need to use both GET and POST requests
@application.route('/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''

    # Check if "password" and "email" POST requests exist (user submitted form)
    if (request.method == 'POST' 
        and 'password' in request.form 
        and 'email' in request.form 
        and 'firstname' in request.form 
        and 'lastname' in request.form):
        # Create variables for easy access


        
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        password = request.form['password']
        email = request.form['email']
        username = email


        
        # Hash the password
        hash = password + application.secret_key
        hash = hashlib.sha1(hash.encode())
        hashed_password = hash.hexdigest();
        

        # Check if account exists using MySQL
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM annotators WHERE username = %s', (username,))
        account = cursor.fetchone()

        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not username or not hashed_password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO annotators VALUES (NULL, %s, %s, %s, %s, %s)', (firstname, lastname, email, username, hashed_password, ))
            mysql.connection.commit()
            msg = Markup('You have successfully registered! Please <a href="' + url_for('login') + '">login</a> to proceed.')




    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'

    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)



# http://localhost:5000/home - this will be the home page, only accessible for loggedin users
@application.route('/home')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('home.html', userid=session['id'],username=session['username'], firstname=session['firstname'])

    # User is not loggedin redirect to login page
    return redirect(url_for('login'))



# http://localhost:5000/review - this will be the review page, only accessible for loggedin users (/<int:prog_num>)
@application.route('/review')
def review():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the review page

        annotator_id = session['id']


        prog_num = int(request.args.get('prog_num'))
        comp_num = int(request.args.get('comp_num'))

        if request.args.get('filter'): 
            flag = int(request.args.get('filter')) # [ abs(int(request.args.get('filter')) - 1 ) ]
        else: 
            flag = -1 #[0,1]
    

        filters = [Annotations.annotator_id == session['id']]
        if flag >= 0:
            filters.append(Annotations.review_flag == flag)

        inprogress = Annotations.query\
                                .join(AllSentences, Annotations.sentence_id == AllSentences.id)\
                                .add_columns(AllSentences.text, 
                                             AllSentences.technique,
                                             Annotations.sentence_id,
                                             Annotations.id, 
                                             Annotations.review_flag,
                                             Annotations.completed)\
                                .filter(Annotations.completed == None)\
                                .filter(*filters)\
                                .paginate(page=prog_num,per_page=10,error_out=False)

        completed = Annotations.query\
                                .join(AllSentences, Annotations.sentence_id == AllSentences.id)\
                                .add_columns(AllSentences.text, 
                                             AllSentences.technique,
                                             Annotations.sentence_id,
                                             Annotations.id, 
                                             Annotations.review_flag,
                                             Annotations.completed)\
                                .filter(Annotations.completed == 1)\
                                .filter(*filters)\
                                .paginate(page=comp_num,per_page=10,error_out=False)

                                #.filter(Annotations.review_flag.in_(flag))\


        meta_data = {'in_progress':inprogress.total,
                     'completed':completed.total}

        
        return render_template('review.html', 
                                userid=session['id'], username=session['username'], firstname=session['firstname'],
                                meta_data=meta_data, 
                                inprogress=inprogress,
                                completed=completed,
                                flag=flag)


    # User is not loggedin redirect to login page
    return redirect(url_for('login'))


# http://localhost:5000/help - this will be the instructions page, only accessible for loggedin users
@application.route('/help')
def help():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the help page
        return render_template('help.html', userid=session['id'], username=session['username'], firstname=session['firstname'])

    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

# http://localhost:5000/references - this will be the instructions page, only accessible for loggedin users
@application.route('/references')
def references():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the references page
        return render_template('references.html', userid=session['id'], username=session['username'], firstname=session['firstname'])

    # User is not loggedin redirect to login page
    return redirect(url_for('login'))


# doing it this way so we can use template {{}} inside javascript
@application.route("/main_js")
def main_js():
    return render_template("/js/main.js")


def tech_to_list(s):
    return [int(i) for i in s[1:-1].split(" ") if len(i)>0]



@application.route("/api/v1/get_stats", methods=["OPTIONS","GET","POST"])
def get_stats():
    response = {"error":"no data"}
    request_data = request.get_json()

    if request_data:
        annotator_id = request_data['annotator_id']

        # query = '''SELECT COUNT(*) FROM annotations WHERE annotator_id=%s AND completed is NULL'''


        query = '''
        SELECT 
          COUNT( CASE WHEN completed is NULL THEN 1 END ) AS t1,
          COUNT( CASE WHEN completed=1 THEN 1 END ) AS t2
        FROM annotations WHERE annotator_id=%s
        '''

        cursor = mysql.connection.cursor()

        cursor.execute(query, (annotator_id, ))
        rows = cursor.fetchall() # store the result before closing the cursor
        mysql.connection.commit()
        cursor.close()


        print("rows",rows)
        response = json.dumps(rows)

    return response




@application.route("/api/v1/get_sentences_from_table", methods=["OPTIONS","GET","POST"])
def get_sentences_from_table():

    next_sentence = {"error":"no data"}
    json_data = request.get_json()
    
    
    if json_data:

        sentence_id = json_data["sentence_id"]
        annotator_id = json_data['annotator_id']

        if "direction" in json_data and json_data["direction"] == "next":
            query = '''
                    SELECT * FROM all_sentences as s
                    LEFT JOIN (SELECT * FROM annotations WHERE annotator_id=%s) as a 
                    ON s.id = a.sentence_id
                    WHERE s.id > %s AND a.completed is NULL ORDER BY s.id ASC LIMIT 1;
                    '''

        elif "direction" in json_data and json_data["direction"] == "prev":
            query = '''
                    SELECT * FROM all_sentences as s
                    LEFT JOIN (SELECT * FROM annotations WHERE annotator_id=%s) as a 
                    ON s.id = a.sentence_id
                    WHERE s.id < %s AND a.completed is NULL ORDER BY s.id DESC LIMIT 1;
                    '''
        else:            
            query = '''
                    SELECT * FROM all_sentences as s
                    LEFT JOIN (SELECT * FROM annotations WHERE annotator_id=%s) as a 
                    ON s.id = a.sentence_id
                    WHERE s.id = %s;
                    '''


        cursor = mysql.connection.cursor()

        cursor.execute(query, (annotator_id,sentence_id,))
        rows = cursor.fetchall() # store the result before closing the cursor
        mysql.connection.commit()
        cursor.close()

        ###### close when finished with the DB ##############################

        sentences_json = []

        for row in rows:
            sentences_json.append({
                "sentence_id" : row[0],
                "article_id" : row[1],
                "technique" : tech_to_list(row[2]),
                "offsets" : row[3],
                "label" : row[4],
                "text" : row[5],
                "parse_string" : row[6],
                "annotation" : row[10] or "",
                "annotator_id" : annotator_id, #row[11] or 0,
                "annotation_id" : row[9] or 0,
                "review_flag" : row[13]
                })
            print('annotation_id',row[9])
        next_sentence =  json.dumps(sentences_json) #jsonify(sentences_json)
        print("next_sentence",next_sentence)
    return next_sentence



@application.route("/api/v1/save_annotation", methods=["OPTIONS","POST"])
def save_annotation():
    
    request_json = request.get_json()


    # remove the following properties:
    properties = ["isCurrNode","depth","x","y","id","x0","y0"]

    if request_json:
        def removeProps(d): 
            for p in properties:
                if p in d:
                    if p == "isCurrNode":
                        d["isCurrNode"] = 0
                    else:
                        del d[p]
            if "children" in d:
                for child in d["children"]:
                    removeProps(child)

        removeProps(request_json)


        ###### open connection to the DB ##########################################
        cursor = mysql.connection.cursor()

        
        sentence_id   = request_json['sentence_id']
        annotation_id = request_json['annotation_id']
        annotator_id  = request_json['annotator_id']
           
        date = time.strftime('%Y-%m-%d %H:%M:%S')
        



        query = '''INSERT INTO annotations (id, json_string, annotator_id, sentence_id, date_updated) VALUES (%s,%s,%s,%s,%s)
          ON DUPLICATE KEY UPDATE json_string=%s, annotator_id=%s, sentence_id=%s, date_updated=%s;'''


        cursor.execute(query, (annotation_id, json.dumps(request_json), annotator_id, sentence_id, date, 
                                json.dumps(request_json), annotator_id, sentence_id, date)) 
        
        mysql.connection.commit()
        print("cursor.lastrowid",cursor.lastrowid)

        if annotation_id == 0:
            annotation_id = cursor.lastrowid

        cursor.close()
        ###### close when finished with the DB ##############################


    fetchDBresult = jsonify(annotation_id)

    return fetchDBresult
    

@application.route("/api/v1/mark_completed", methods=["OPTIONS","POST"])
def mark_completed():
    response = {"error":"no data"}
    request_data = request.get_json()

    if request_data:
        annotation_id = request_data['annotation_id']
        
        print("annotation_id",annotation_id)

        query = '''UPDATE annotations SET completed=1 WHERE id=%s'''

        cursor = mysql.connection.cursor()

        cursor.execute(query, (annotation_id,))
        rows = cursor.fetchall() # store the result before closing the cursor
        mysql.connection.commit()
        cursor.close()

        response = json.dumps(rows)

    return json.dumps({"asnswer":"marked completed"})




@application.route("/api/v1/flag_for_review", methods=["OPTIONS","POST"])
def flag_for_review():
    response = {"error":"no data"}
    request_data = request.get_json()

    if request_data:
        annotation_id = request_data['annotation_id']
        
        print("annotation_id",annotation_id)

        get_flag = '''SELECT review_flag FROM annotations WHERE id=%s'''

        set_flag = '''UPDATE annotations SET review_flag=%s WHERE id=%s'''

        cursor = mysql.connection.cursor()

        cursor.execute(get_flag, (annotation_id,))
        flag = cursor.fetchall()[0][0]
        flag = abs(flag-1)

        cursor.execute(set_flag, (flag,annotation_id,))
        rows = cursor.fetchall() # store the result before closing the cursor
        mysql.connection.commit()
        cursor.close()

        response = json.dumps(rows)

    return json.dumps({"review_flag":flag})


################################################################################################################################

# load features for ChatGPT
sentenceFeaturesDict = defaultdict(list)
sentenceFeaturesList = []

with open('features-sentence.json') as f:
    for jsonObj in f:
        featureDict = json.loads(jsonObj)
        sentenceFeaturesList.append(featureDict)
        sentenceFeaturesDict[featureDict['key']] = [val+" : "+desc for val, desc in zip(featureDict['values'],featureDict['descriptions'])]


wordFeaturesDict = defaultdict(list)
wordFeaturesList = []

with open('features-word.json') as f:
    for jsonObj in f:
        featureDict = json.loads(jsonObj)
        wordFeaturesList.append(featureDict)
        wordFeaturesDict[featureDict['key']] = [val+" : "+desc for val, desc in zip(featureDict['values'],featureDict['descriptions'])]



@application.route("/api/v1/get_word_properties", methods=["OPTIONS","GET","POST"])
def get_word_properties():
    return json.dumps(wordFeaturesList)

@application.route("/api/v1/get_sentence_properties", methods=["OPTIONS","GET","POST"])
def get_sentence_properties():
    return json.dumps(sentenceFeaturesList)


featuresDict = sentenceFeaturesDict | wordFeaturesDict


@application.route("/api/v1/get_gpt_response", methods=["OPTIONS","POST"])
def get_gpt_response():
    
    response = {"error":"no data"}
    json_data = request.get_json()


    
    if json_data:
        sentence = json_data['sentence']
        feature = json_data['feature']
        # prompt = json_data['prompt']

        properties = "\n".join(featuresDict[feature])

        prompt = f"""
        You are a rhetoretician and linguist specializing in news text. 

        Your task is to identify which, if any, of the following properties of {feature} are used in the example text. 
        You may select multiple properties.
        Each line contains a property followed by a colon, followed by a brief definition and example(s):
        
        {properties}

        
        Format your response as a JSON object with "Property" and "Explanation" as the keys. 
        If the property isn't present, use "unknown" as the value of "Property". 
        Explain your choise in the "Explanation". Make your response as short as possible.
        The example text is delimited with triple backticks. 
        
          
        Example text: ```{sentence}```
        """

        openai.organization = os.getenv("OPENAI_ORG_ID")
        openai.api_key = os.getenv("OPENAI_API_KEY")

        model = "gpt-3.5-turbo-0301"
        model = "gpt-3.5-turbo-0613"

                    
        messages=[{"role": "user", "content": prompt}]
        res = openai.ChatCompletion.create(
          model=model,
          messages=messages
        )
        

        response = res["choices"][0]["message"]["content"]

    return response

# run the application.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production application.
    application.debug = True
    application.run(port=5002)