from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
application = Flask(__name__, static_folder='static')
app = application

@application.route('/',methods=['GET','POST'])
@cross_origin() # its purpose is to be available to different countries
def index():
    return render_template("landing.html")
  
@application.route('/gpt',methods=['GET','POST'])
@cross_origin() # its purpose is to be available to different countries
def index1():
    import os
#     import openai
#     openai.api_key ="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"
    input_text = request.form['input-field']
#     completion = openai.ChatCompletion.create(
#       model="gpt-3.5-turbo",
#       messages=[
#         {"role": "user", "content": input_text}
#       ]
#     )
    import cohere

# Set up a Cohere API client with your API key
    client = cohere.Client("5dPAagesHxYyp8kIg9QIyxknJB7WlAKZJmAmHXwJ")

# Define the text you want to summarize
    text = f'''{input_text}     explain it in minimum 1000 words                                                                                                                                                                                                                                                                                                               .'''

# Call the summarize method with your text
    summary = client.summarize(text) #, num_sentences=2
  
# Print the summary
    output_text = summary.summary
    # output_text=completion.choices[0].message['content']

#     output_text=completion.choices[0].message['content']
    return render_template('landing.html',output=output_text)

@application.route('/subscribe',methods=['GET','POST'])
@cross_origin() # its purpose is to be available to different countries
def index3():
      import pymongo
      email = request.form['email']
      
      client = pymongo.MongoClient("mongodb+srv://breakratr:breakratr@vivekdb.fwdld9x.mongodb.net/?retryWrites=true&w=majority")
      db = client['WeCare']
      collection_1 = db['Subscription']
      dict = {'Email':email}
      collection_1.insert_one(dict)
      return render_template("landing.html")
    
@application.route('/contact',methods=['GET','POST'])
@cross_origin()
def index4():
      return render_template("contact_us.html")
    
@application.route('/contacts',methods=['GET','POST'])
@cross_origin() # its purpose is to be available to different countries
def index5():
      import pymongo
      name = request.form['name']
      email = request.form['email']
      message = request.form['message']
      client = pymongo.MongoClient("mongodb+srv://breakratr:breakratr@vivekdb.fwdld9x.mongodb.net/?retryWrites=true&w=majority")
      db = client['WeCare']
      collection_1 = db['contact']
      dict = {'Name':name,'Email':email,'Message':message}
      collection_1.insert_one(dict)
      return render_template("landing.html")

@application.route('/appoint',methods=['GET','POST'])
@cross_origin()
def index6():
      return render_template("Appointment(1).html")
    
@application.route('/appointed',methods=['GET','POST'])
@cross_origin()
def index8():
  import pymongo
  doc_name = request.form['doc_name']
  email = request.form['email']
  issue = request.form['issue']
  mob = request.form['mob']
  name = request.form['name']
  client = pymongo.MongoClient("mongodb+srv://breakratr:breakratr@vivekdb.fwdld9x.mongodb.net/?retryWrites=true&w=majority")
  db = client['WeCare']
  collection_1 = db['Appointment']
  dict = {'Name':name,'Email':email,'Doctor name ':doc_name,'Issue':issue,'Mobile Number ':mob}
  collection_1.insert_one(dict)
  return render_template("appointmentSuccess.html")

@application.route('/aboutUs',methods=['GET','POST'])
@cross_origin()
def index7():
      return render_template("aboutUs.html")
    
if __name__ == '__main__':
    application.run(debug=True)
