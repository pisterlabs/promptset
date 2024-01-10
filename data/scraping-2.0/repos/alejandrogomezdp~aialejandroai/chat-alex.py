from flask import Flask, request, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from nltk import sent_tokenize, word_tokenize, pos_tag
from bs4 import BeautifulSoup
import openai

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite3"
db = SQLAlchemy(app)

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_input = db.Column(db.String(500))
    chatbot_response = db.Column(db.String(5000))

    def __init__(self, user_input, chatbot_response):
        self.user_input = user_input
        self.chatbot_response = chatbot_response


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(5000))
    role = db.Column(db.String(10))
    conversation_id = db.Column(db.Integer, db.ForeignKey("conversation.id"))

    def __init__(self, content, role, conversation_id):
        self.content = content
        self.role = role
        self.conversation_id = conversation_id


openai.api_key = "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"

def format_text(text):
    sentences = sent_tokenize(text)

    formatted_text = ""
    for sentence in sentences:
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)

        formatted_sentence = ""
        for word, tag in tagged_words:
            if tag.startswith("NN"):
                formatted_sentence += f"<b>{word}</b> "
            elif tag.startswith("VB"):
                formatted_sentence += f"<i>{word}</i> "
            else:
                formatted_sentence += f"{word} "
        
        formatted_text += f"<p>{formatted_sentence}</p>"

    soup = BeautifulSoup(formatted_text, "html.parser")
    formatted_text = soup.prettify()

    return formatted_text


@app.route("/chat", methods=["GET", "POST"])
def chat():
    messages = []
    if request.method == "POST":
        user_input = request.json.get("message")
        if user_input:
            conv = Conversation(user_input, "")
            db.session.add(conv)
            db.session.commit()
            user_msg = Message(content=user_input, role="user", conversation_id=conv.id)
            db.session.add(user_msg)
            db.session.commit()

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un experto en programación reconocido y tutor con 10 años de codificación. Estás ayudando a un profesional con un problema de codificación.",
                    },
                    {"role": "user", "content": user_input},
                ],
            )

            response = response.choices[0].message["content"]
            bot_msg = Message(content=format_text(response), role="bot", conversation_id=conv.id)
            db.session.add(bot_msg)
            db.session.commit()

        messages = Message.query.filter_by(conversation_id=conv.id).all()
        return response

    convs = Conversation.query.all()
    return render_template("chat.html", conversations=convs, messages=messages)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)

