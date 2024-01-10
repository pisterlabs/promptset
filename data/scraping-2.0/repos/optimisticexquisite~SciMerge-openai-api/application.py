import openai
import uuid
from datetime import timedelta
import pymongo
import hashlib
from scholarly import scholarly
from flask import Flask, render_template, request, redirect, url_for, session
from flask import jsonify
import json
app = Flask(__name__, static_folder="static")
app.secret_key = 'mysecretkey368768uyfj9vu86fy'
app.config['PERMANENT_SESSION_LIFETIME'] =  timedelta(minutes=120)
app.config['FLASK_APP'] = app
app.config['TEMPLATES_AUTO_RELOAD'] = True
openai.api_key='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'
m=hashlib.sha256()
myclient=pymongo.MongoClient("mongodb+srv://scimerge358:abcd1234@cluster0.ehalt0r.mongodb.net/")
scimerge=myclient["scimerge"]
users=scimerge["users"]
data=scimerge["data"]
@app.route('/api/liveprompt',methods=['POST'])
def liveprompt():
    received_data=request.get_json()
    username=received_data['username']
    title=received_data['title']
    abstract=received_data['abstract']
    question="Title:'"+title+"'\n\nAbstract: "+abstract
    engineeredprompt=question+'\n'+"Make points which can be added to this abstract to make it more informative and useful for the reader.\n\n Response: MUST BE JSON(point1: 'point1', point2: 'point2', point3: 'point3', point4: 'point4', point5: 'point5')"
    questionreply=openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k-0613",
            temperature=0,

            messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": f"{engineeredprompt}"}
            ],       
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
    i=1
    points=[]
    while True:
            replytext=questionreply.choices[0].message.content
            points.append(json.loads(replytext)[f"point{i}"])
            print(points[i-1])
            if i==5:
                break
            i+=1
    return jsonify(points)
    # return questionreply.choices[0].message.content
@app.route('/api/tags/profile',methods=['POST'])
def profiletags():
    received_data=request.get_json()
    username=received_data['username']
    explainedinterests=received_data['explainedinterests']
    previousexperience=received_data['previousexperience']
    question="Interests: "+explainedinterests+"\n\nPrevious Experience: "+previousexperience
    engineeredprompt=question+'\n'+"Make EXACTLY 20 tags which can be added to this profile according to the specified interests and previous experiences\n\n Response: MUST BE JSON(tag1: 'tag1', tag2: 'tag2', tag3: 'tag3', tag4: 'tag4', tag5: 'tag5', tag6: 'tag6', tag7: 'tag7', tag8: 'tag8', tag9: 'tag9', tag10: 'tag10', tag11: 'tag11', tag12: 'tag12', tag13: 'tag13', tag14: 'tag14', tag15: 'tag15', tag16: 'tag16', tag17: 'tag17', tag18: 'tag18', tag19: 'tag19', tag20: 'tag20')"
    questionreply=openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k-0613",
            temperature=1,

            messages=[
                    {"role": "system", "content": "You're supposed to be creative while STRICTLY following instructions."},
                    {"role": "user", "content": f"{engineeredprompt}"}
            ],       
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
    i=1
    points=[]
    while True:
        replytext=questionreply.choices[0].message.content
        try:
            points.append(json.loads(replytext)[f"tag{i}"])
            print(points[i-1])
        except:
             pass
        if i==20:
            break
        i+=1

    return jsonify(points)

@app.route('/api/tags/project',methods=['POST'])
def projecttag():
    received_data=request.get_json()
    username=received_data['username']
    title=received_data['title']
    abstract=received_data['abstract']
    question="Title:'"+title+"'\n\nAbstract: "+abstract
    engineeredprompt=question+'\n'+"Make EXACTLY 20 tags which can be added to this project according to the specified title and abstract\n\n Response: MUST BE JSON(tag1: 'tag1', tag2: 'tag2', tag3: 'tag3', tag4: 'tag4', tag5: 'tag5', tag6: 'tag6', tag7: 'tag7', tag8: 'tag8', tag9: 'tag9', tag10: 'tag10', tag11: 'tag11', tag12: 'tag12', tag13: 'tag13', tag14: 'tag14', tag15: 'tag15', tag16: 'tag16', tag17: 'tag17', tag18: 'tag18', tag19: 'tag19', tag20: 'tag20')"
    questionreply=openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k-0613",
            temperature=1,

            messages=[
                    {"role": "system", "content": "You're supposed to be creative while STRICTLY following instructions."},
                    {"role": "user", "content": f"{engineeredprompt}"}
            ],       
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
    i=1
    points=[]
    while True:
        replytext=questionreply.choices[0].message.content
        try:
            points.append(json.loads(replytext)[f"tag{i}"])
            print(points[i-1])
        except:
             pass
        if i==20:
            break
        i+=1

    return jsonify(points)

@app.route('/api/scholarly',methods=['POST'])
def scholarlyapi():
    receiveddata=request.get_json()
    username=receiveddata['username']
    title=receiveddata['title']
    search_query = scholarly.search_pubs(f"Title:'+{title}")
    jsondata=[]
    for i in search_query:
        jsondata.append(i)
    return jsonify(jsondata)




#WEB APIs
@app.route('/webapi/project/<projectid>',methods=['GET'])
def projectwebapi(projectid):
    project=data.find_one({"uniqueid":projectid})
    jsonfile={}
    jsonfile['title']=project['title']
    jsonfile['abstract']=project['abstract']
    jsonfile['tags']=project['tags']
    jsonfile['username']=project['username']
    jsonfile['uniqueid']=project['uniqueid']
    return jsonify(jsonfile)














@app.route('/',methods=['GET'])
def index():
    if 'username' in session:
        return redirect(url_for('home'))
    else:
        return redirect(url_for('login'))


@app.route('/login',methods=['GET','POST'])
def login():
    if request.method=='POST':
        m=hashlib.sha256()
        m.update(request.form['password'].encode('utf-8'))
        password=m.hexdigest()
        user=users.find_one({"username":request.form['username']})
        if user:
            if user['password']==password:
                session['username']=user['username']
                return redirect(url_for('home'))
            else:
                return render_template('login.html')
    return render_template('login.html')

@app.route('/logout',methods=['GET','POST'])
def logout():
    session.pop('username',None)
    return redirect(url_for('login'))

@app.route('/home',methods=['GET','POST'])
def home():
    if request.method=='POST':
          print("Posted")
    if 'username' in session:
        user=users.find_one({"username":session['username']})
        currentprojects=json.dumps(user['currentprojects'])
        pastprojects=json.dumps(user['pastprojects'])
        return render_template('index.html',userdata=user,currentprojects=currentprojects,pastprojects=pastprojects)
    else:
        return render_template('login.html')
    
@app.route('/createproject',methods=['GET','POST'])
def createproject():
    if 'username' in session:
        if request.method=='POST':
            if 'username' in session:
                jsondata=request.get_json()
                title=jsondata['title']
                abstract=jsondata['abstract']
                tags=jsondata['tags']
                uniqueid=str(uuid.uuid4())
                data.insert_one({"username":session['username'],"title":title,"abstract":abstract,"tags":tags,"uniqueid":uniqueid})
                users.update_one({"username":session['username']},{"$push":{"currentprojects":uniqueid}})
                return jsonify({"uniqueid":uniqueid})
            else:
                return render_template('createproject.html')
        return render_template('createproject.html')
    else:
        return redirect(url_for('home'))

@app.route('/suggestions/<projectid>',methods=['GET','POST'])
def suggestions(projectid):
    if 'username' in session:
        if request.method=='POST':
            print("Posted")
        project=data.find_one({"uniqueid":projectid})
        return render_template('suggestions.html',project=project)
    else:
        return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)