from cheapsharkapi import return_cheapest as rc
from cheapsharkapi import return_game as rg
from cheapsharkapi import set_alert as sa
from cheapsharkapi import manage_alerts as ma
from flask import Flask, request, render_template, session
import os
import openai
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import json

OPENAI_KEY = os.getenv("OPENAI_KEY")
openai.api_key = OPENAI_KEY

# https://apidocs.cheapshark.com/

SQLPASS = os.getenv("SQLPASS")
CHEAPSHARK_API_DEALS = "https://www.cheapshark.com/api/1.0/deals"
CHEAPSHARK_API_STORES = "https://www.cheapshark.com/api/1.0/stores"
CHEAPSHARP_REDIRECT = "https://www.cheapshark.com/redirect?dealID="
CHEAPSHARK_API_ALERT = "https://www.cheapshark.com/api/1.0/alerts"

db = SQLAlchemy()
app = Flask(__name__)

SQLALCHEMY_DATABASE_URI = "mysql+mysqlconnector://{username}:{password}@{hostname}/{databasename}?collation=utf8mb4_general_ci".format(
    username="rnicosia",
    password=SQLPASS,
    hostname="rnicosia.mysql.pythonanywhere-services.com",
    databasename="rnicosia$conversations",
)
app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
app.config["SQLALCHEMY_POOL_RECYCLE"] = 299
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

migrate = Migrate(app, db)

input_data = ''


class Conversation(db.Model):
    id = db.Column(db.String(256), primary_key=True)
    messages = db.Column(db.Text)  # Store conversation messages as a JSON string


"""
try:
    with app.app_context():
        db.create_all()
except Exception as e:
    print(e)
"""


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("main_page.html")
    elif request.method == "POST":
        if request.form['get_games'] == "Free Games":
            deals_params = {
                "sortBy": "Price",
                "pageNumber": 0,
                "lowerPrice": 0,
                "upperPrice": 0,
            }
            cheapshark_data = rc(deals_params, CHEAPSHARK_API_DEALS, CHEAPSHARK_API_STORES, CHEAPSHARP_REDIRECT)
            return render_template("main_page.html", cheapshark_data=cheapshark_data)
        elif request.form['get_games'] == "Games Under $10":
            deals_params = {
                "sortBy": "Price",
                "pageNumber": 0,
                "lowerPrice": 0.01,
                "upperPrice": 10,
            }
            cheapshark_data = rc(deals_params, CHEAPSHARK_API_DEALS, CHEAPSHARK_API_STORES, CHEAPSHARP_REDIRECT)
            return render_template("main_page.html", cheapshark_data=cheapshark_data)
        elif request.form['get_games'] == "Games $10 to $20":
            deals_params = {
                "sortBy": "Price",
                "pageNumber": 0,
                "lowerPrice": 10,
                "upperPrice": 20,
            }
            cheapshark_data = rc(deals_params, CHEAPSHARK_API_DEALS, CHEAPSHARK_API_STORES, CHEAPSHARP_REDIRECT)
            return render_template("main_page.html", cheapshark_data=cheapshark_data)
        elif request.form['get_games'] == "Games $20 to $30":
            deals_params = {
                "sortBy": "Price",
                "pageNumber": 0,
                "lowerPrice": 20,
                "upperPrice": 30,
            }
            cheapshark_data = rc(deals_params, CHEAPSHARK_API_DEALS, CHEAPSHARK_API_STORES, CHEAPSHARP_REDIRECT)
            return render_template("main_page.html", cheapshark_data=cheapshark_data)
        elif request.form['get_games'] == "Games $30 to $40":
            deals_params = {
                "sortBy": "Price",
                "pageNumber": 0,
                "lowerPrice": 30,
                "upperPrice": 40,
            }
            cheapshark_data = rc(deals_params, CHEAPSHARK_API_DEALS, CHEAPSHARK_API_STORES, CHEAPSHARP_REDIRECT)
            return render_template("main_page.html", cheapshark_data=cheapshark_data)
        elif request.form['get_games'] == "Games $40 to $50":
            deals_params = {
                "sortBy": "Price",
                "pageNumber": 0,
                "lowerPrice": 40,
                "upperPrice": 49.99,
            }
            cheapshark_data = rc(deals_params, CHEAPSHARK_API_DEALS, CHEAPSHARK_API_STORES, CHEAPSHARP_REDIRECT)
            return render_template("main_page.html", cheapshark_data=cheapshark_data)
        elif request.form['get_games'] == "Games Over $50":
            deals_params = {
                "sortBy": "Price",
                "pageNumber": 0,
                "lowerPrice": 50,
                "upperPrice": 50,
            }
            cheapshark_data = rc(deals_params, CHEAPSHARK_API_DEALS, CHEAPSHARK_API_STORES, CHEAPSHARP_REDIRECT)
            return render_template("main_page.html", cheapshark_data=cheapshark_data)
        elif request.form['get_games'] == "Search":
            input_data = request.form['game_title_text']
            deals_params = {
                "sortBy": "Price",
                "title": input_data
            }
            cheapshark_data = rg(deals_params, CHEAPSHARK_API_DEALS, CHEAPSHARK_API_STORES, CHEAPSHARP_REDIRECT)
            return render_template("main_page.html", cheapshark_data=cheapshark_data)
        elif request.form['get_games'] == "Set Alert":
            price = request.form['price']
            email = request.form['email']
            gameid = request.form['gameid']
            alert_params = {
                "action": "set",
                "price": price,
                "email": email,
                "gameID": gameid
            }
            cheapshark_data = sa(CHEAPSHARK_API_ALERT, alert_params)
            return render_template("main_page.html", cheapshark_data=cheapshark_data)
        elif request.form['get_games'] == "Delete Alert":
            price = request.form['price']
            email = request.form['email']
            gameid = request.form['gameid']
            alert_params = {
                "action": "delete",
                "price": price,
                "email": email,
                "gameID": gameid
            }
            cheapshark_data = sa(CHEAPSHARK_API_ALERT, alert_params)
            return render_template("main_page.html", cheapshark_data=cheapshark_data)
        elif request.form['get_games'] == "Manage Alerts":
            email = request.form['manageemail']
            alert_params = {
                "action": "manage",
                "email": email
            }
            cheapshark_data = ma(CHEAPSHARK_API_ALERT, alert_params)
            return render_template("main_page.html", cheapshark_data=cheapshark_data)
        elif request.form['get_games'] == "Metacritic 90+ Games":
            deals_params = {
                "sortBy": "Price",
                "metacritic": 90
            }
            cheapshark_data = rc(deals_params, CHEAPSHARK_API_DEALS, CHEAPSHARK_API_STORES, CHEAPSHARP_REDIRECT)
            return render_template("main_page.html", cheapshark_data=cheapshark_data)
    # user_input_data.append(request.form["contents"])


@app.route('/fun', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        question = request.form.get('question')
        model = request.form.get('model')
        conversation_id = request.form.get('conversation_id')

        # get existing conversation or start a new one
        conversation_record = Conversation.query.get(conversation_id)
        if conversation_record:
            conversation = json.loads(conversation_record.messages)
        else:
            conversation = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                }
            ]

        conversation.append({
            "role": "user",
            "content": question
        })

        response = openai.ChatCompletion.create(
            model=model,
            messages=conversation
        )

        # add the assistant's reply to the conversation
        conversation.append({
            "role": "assistant",
            "content": response['choices'][0]['message']['content']
        })

        # update the stored conversations
        if conversation_record:
            conversation_record.messages = json.dumps(conversation)
        else:
            conversation_record = Conversation(id=conversation_id, messages=json.dumps(conversation))
            db.session.add(conversation_record)
        db.session.commit()

        conversations = {conversation.id: json.loads(conversation.messages) for conversation in
                         Conversation.query.all()}

        return render_template('ask_gptv2.html', answer=response['choices'][0]['message']['content'],
                               conversations=conversations)
    else:
        conversations = {conversation.id: json.loads(conversation.messages) for conversation in
                         Conversation.query.all()}
        return render_template('ask_gptv2.html', conversations=conversations)


#dive_site_stuff_below
@app.route('/dive-site')
def dive_site_index():
        return render_template('dive-site/home.html')

@app.route('/dive-site/about')
def dive_site_about():
        return render_template('dive-site/about.html')

@app.route('/dive-site/contact')
def dive_site_contact():
        return render_template('dive-site/contact.html')

@app.route('/dive-site/destinations')
def dive_site_destinations():
        return render_template('dive-site/destinations.html')

@app.route('/dive-site/indonesia')
def dive_site_indonesia():
        return render_template('dive-site/indonesia.html')

@app.route('/dive-site/maldives')
def dive_site_maldives():
        return render_template('dive-site/maldives.html')

@app.route('/dive-site/galapagos')
def dive_site_galapagos():
        return render_template('dive-site/galapagos.html')

@app.route('/dive-site/french-polynesia')
def dive_site_french_polynesia():
        return render_template('dive-site/french-polynesia.html')

@app.route('/dive-site/papua-new-guinea')
def dive_site_papua_new_guinea():
        return render_template('dive-site/papua-new-guinea.html')

@app.route('/dive-site/fiji')
def dive_site_fiji():
        return render_template('dive-site/fiji.html')

if __name__ == '__main__':
    app.run(debug=False)
