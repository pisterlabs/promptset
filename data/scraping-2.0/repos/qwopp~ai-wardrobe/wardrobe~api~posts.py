"""REST API for wardrobe."""

import hashlib
import uuid
import flask
import wardrobe
import openai
openai.api_type = "azure"
openai.api_key = '07c5f40ba37b4b40863a80101eaf2105'
openai.api_base = 'https://api.umgpt.umich.edu/azure-openai-api/ptu'
openai.api_version = '2023-03-15-preview'
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import os


def get_most_recent_clothesid(logname):
    """Return most recent clothesid by user"""
    connection = wardrobe.model.get_db()
    cur = connection.execute(
        """
        SELECT clothing.clothesid
        FROM clothing
        WHERE clothing.owner = ? 
        ORDER BY clothing.clothesid DESC
        LIMIT 1
        """,
        (logname,),
    )
    m_r = cur.fetchone()
    if m_r:
        return m_r['clothesid']
    return 0


def not_logged():
    """Return 403 if user is not logged in."""
    print("WE AINT LOGGED IN!")
    context1 = {
        "message": "Forbidden",
        "status_code": 403
    }
    return flask.jsonify(**context1), 403


def verify_pass(password, storedhash):
    """Wardrobe check if password matches."""
    algo, salt2, passhash = storedhash.split('$')
    hash_obj58 = hashlib.new(algo)
    saltedpass = salt2 + password
    hash_obj58.update(saltedpass.encode('utf-8'))
    print(passhash)
    print(hash_obj58.hexdigest())
    return passhash == hash_obj58.hexdigest()


def check_logged():
    """Check if user is logged in."""
    logname = flask.session.get('username')
    if 'username' not in flask.session:
        auth = flask.request.authorization
        if auth is not None and 'username' in auth and 'password' in auth:
            username = flask.request.authorization['username']
            password = flask.request.authorization['password']
        else:
            return "notloggedin"
        connection = wardrobe.model.get_db()
        if not username or not password:
            return "notloggedin"
        cur = connection.execute(
            """
            SELECT username, password FROM users WHERE username = ?
            """,
            (username,),
        )
        user = cur.fetchone()
        if user and verify_pass(password, user['password']):
            flask.session['username'] = user['username']
            logname = user['username']
        else:
            return "notloggedin"
    return logname


@wardrobe.app.route('/api/v1/', methods=["GET"])
def get_api():
    """Return API services."""
    context = {
        "clothes": "/api/v1/clothes/",
        "url": flask.request.path,
    }
    return flask.jsonify(**context)


@wardrobe.app.route('/api/v1/clothing/', methods=["GET"])
def get_clothing():
    """Return the clothes stored on the logged-in user's account."""
    logname = check_logged()
    if logname == "notloggedin":
        print("NOT LOGGED IN")
        return not_logged()
    
    size = flask.request.args.get('size', default=12, type=int)
    page = flask.request.args.get('page', default=0, type=int)
    m_r = get_most_recent_clothesid(logname)
    clothesid_lte = flask.request.args.get('clothesid_lte', default=m_r, type=int)
    newp2 = flask.request.full_path
    if newp2.endswith('?'):
        newp2 = newp2[:-1]
    if size <= 0 or page < 0 or (clothesid_lte is not None and clothesid_lte < 0):
        context = {
            "message": "Bad Request",
            "status_code": 400
        }
        return flask.jsonify(**context), 400

    connection = wardrobe.model.get_db()
    cur = connection.execute(
        """
        SELECT clothing.clothesid, clothing.filename, clothing.owner, 
        clothing.article, clothing.confidence
        FROM clothing
        WHERE clothing.owner = ?
        AND (clothing.clothesid <= ? OR ? IS NULL)
        ORDER BY clothing.clothesid DESC
        LIMIT ? OFFSET ?
        """,
        (logname, clothesid_lte, clothesid_lte, size, page * size),
    )
    clothes_data = cur.fetchall()
    
    next_page_url = ""
    if len(clothes_data) >= size:
        npu = "/api/v1/clothing/?size={}&page={}&clothesid_lte={}"
        next_page_url = npu.format(size, page + 1, clothesid_lte)
    response = {
        "next": next_page_url,
        "results": [
            {
                "clothesid": clothing['clothesid'],
                "filename": "/uploads/" + clothing['filename'],
                "owner": clothing['owner'],
                "article": clothing['article'],
                "confidence": clothing['confidence'],
                "url": f"/api/v1/clothing/{clothing['clothesid']}/"
            }
            for clothing in clothes_data
        ],
        "url": newp2,
    }
    return flask.jsonify(**response)


@wardrobe.app.route('/api/v1/prompt/', methods=["POST"])
def prompt_to_output():
    logname = check_logged()
    data = flask.request.json
    prompt = data.get('inputData')
    # Get prompt input
    # Get all articles from DB
    connection = wardrobe.model.get_db()
    cur = connection.execute(
        """
        SELECT clothing.article
        FROM clothing
        WHERE clothing.owner = ?
        ORDER BY clothing.clothesid DESC;
        """,
        (logname,),
    )
    dats2 = cur.fetchall()
    dats_prompt = "List of clothing in closet: "
    gpt_prompt = "Using the list given, give me an outfit based off of the following prompt: (" + prompt +  "). You should only return the required pieces of the outfit. You should only include 1 shirt maximum, and 1 shorts maximum. The output should be sorted from head to toe, separated by commas with spaces after the commas. You should under no circumstance include text before or after the outfit."
    gbt_output = ""
    #print(dats2)
    for x in dats2:
        dats_prompt += x['article'] + ","
    #print(dats_prompt)
    #print(gpt_prompt)
    #print(dats_prompt)
    #print(gpt_prompt)
    try:
        response = openai.ChatCompletion.create(
        engine='gpt-4',
        messages=[
            {"role": "system", "content": dats_prompt},
            {"role": "user", "content": gpt_prompt}
        ]
    )
        
        gbt_output = response['choices'][0]['message']['content']
        print(gbt_output)
        # Parse GPT output
        gbt_output = gbt_output.split(",")
        gbt_output = [item.strip() for item in gbt_output]
    except openai.error.APIError as e:
        # Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")

    except openai.error.AuthenticationError as e:
        # Handle Authentication error here, e.g. invalid API key
        print(f"OpenAI API returned an Authentication Error: {e}")

    except openai.error.APIConnectionError as e:
        # Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")

    except openai.error.InvalidRequestError as e:
        # Handle connection error here
        print(f"Invalid Request Error: {e}")

    except openai.error.RateLimitError as e:
        # Handle rate limit error
        print(f"OpenAI API request exceeded rate limit: {e}")

    except openai.error.ServiceUnavailableError as e:
        # Handle Service Unavailable error
        print(f"Service Unavailable: {e}")

    except openai.error.Timeout as e:
        # Handle request timeout
        print(f"Request timed out: {e}")

    except:
        print("An exception has occured.")
    print(gbt_output)
    connection = wardrobe.model.get_db()
    file_image_data = {}
    for article in gbt_output:
        cur = connection.execute(
            """
            SELECT filename
            FROM clothing
            WHERE article = ?
            LIMIT 1
            """,
            (article,)
        )
        files = cur.fetchall()
        file_image_data[article] = [file_data['filename'] for file_data in files]
    ordered_file_image_data = [file_image_data[article] for article in gbt_output]
    ordered_file_image_data_flat = [f"/uploads/{filename}" for sublist in ordered_file_image_data for filename in sublist]
    response_data = {
        "imageFiles": ordered_file_image_data_flat
    }
    return flask.jsonify(response_data), 200


@wardrobe.app.route('/api/v1/upload/', methods=['POST'])
def upload_file():
    logname = check_logged()
    if 'username' not in flask.session:
        flask.abort(403)
    file = flask.request.files['file']
    if file.filename == '':
        return flask.jsonify({"message": "No file selected."}), 400
    img_height = 180
    img_width = 180
    # verify logged in user
    filename = file.filename
    stem = uuid.uuid4().hex
    suffix = pathlib.Path(filename).suffix.lower()
    uuid_basename = f"{stem}{suffix}"
    if not file:
        flask.abort(400)
    path = wardrobe.app.config["UPLOAD_FOLDER"] / uuid_basename
    file.save(path)
    model = tf.keras.models.load_model('my_model_RMSprop.keras')    
    new_image_path = path
    img = tf.keras.utils.load_img(
        new_image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_names = ['black_dress', 'black_pants', 'black_shirt', 'black_shoes', 'black_shorts', 'black_suit', 'blue_dress', 'blue_pants', 'blue_shirt', 'blue_shoes', 'blue_shorts', 'brown_hoodie', 'brown_pants', 'brown_shoes', 'green_pants', 'green_shirt', 'green_shoes', 'green_shorts', 'green_suit', 'pink_hoodie', 'pink_pants', 'pink_skirt', 'red_dress', 'red_hoodie', 'red_pants', 'red_shirt', 'red_shoes', 'silver_shoes', 'silver_skirt', 'white_dress', 'white_pants', 'white_shoes', 'white_shorts', 'white_suit', 'yellow_dress', 'yellow_shorts', 'yellow_skirt']
    article = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    print(article)
    print(confidence)
    connection = wardrobe.model.get_db()
    connection.execute(
        """
        INSERT INTO clothing (filename, owner, article, confidence)
        VALUES (?, ?, ?, ?)
        """,
        (uuid_basename, logname, article, confidence),
    )
    connection.commit()
    return flask.jsonify({"message": "File uploaded successfully."}), 200





