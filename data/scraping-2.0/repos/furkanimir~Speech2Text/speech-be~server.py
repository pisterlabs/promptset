from flask import jsonify
from flask import Flask
from flask import request
import openai
openai.api_key = 'BURAYA API GIRINIZ'

conten1 = 'what is a difference between npx and npm'
conten2 = 'which lol hero should support for Jinx'
conten3 = 'how many moons does jupiter have'
conten4 = "how many kinds of soup are there in turkey?"

# content =[{}]
def aloGpt(aloalo):
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "user", "content": aloalo}
      ]
    )

    chat_response = completion.choices[0].message
    return chat_response.content

app = Flask(__name__)




degi =""

@app.route('/api/data', methods=['GET'])
def get_data():
    data = {'chat_response': aloGpt(degi)}
    print(degi)
    return jsonify(data)


@app.route('/api/post', methods = ['POST'])
def update_text():
    data = request.json
    if data is None:
        return 'Hatalı istek', 400
    global degi
    degi=data['text']
    print(degi)
    return degi


if __name__ == "__main__":
    app.run(debug=True)






# @app.route("/flask_api")
# def flask_api():
#    # Creating the dictionary
#    return jsonify({
#       "chat_response" : aloGpt(conten3)
#    })

# @app.route("/flask_api2")
# def flask_api2():
#
#    return jsonify({
#     "name": "John Smith",
#     "age": 30,
#     "address": {
#         "street": "123 Main St",
#         "city": "Anytown",
#         "state": "USA"
#     },
#     "message": "It's a lovely day"
# })