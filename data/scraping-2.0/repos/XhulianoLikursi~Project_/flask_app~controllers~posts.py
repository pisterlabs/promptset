from flask_app import app
from flask_app.models.user import User
from flask_app.models.post import Post



from flask import render_template, redirect, session, request, flash, jsonify

from .env import UPLOAD_FOLDER
from .env import ALLOWED_EXTENSIONS
from datetime import datetime
from werkzeug.utils import secure_filename
import os
import openai 




app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
# The limit is 3 MB


# Set up your OpenAI API credentials
openai.api_key = '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'


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/add/post')
def addPost():
    if 'user_id' in session:
        data = {
            'user_id': session['user_id']
        }
        loggedUser = User.get_user_by_id(data)
        return render_template('addPost.html', loggedUser = loggedUser)
    return redirect('/')

@app.route('/create/post', methods = ['POST'])
def createPost():
    if 'user_id' in session:
        if not Post.validate_post(request.form):
            return redirect(request.referrer)
        

     
        if not request.files['image']:
            flash('Image is required!', 'post')
            return redirect(request.referrer)
        image = request.files['image']

        if not allowed_file(image.filename):
            flash('Image should be in png, jpg. jpeg format!', 'postImage')
            return redirect(request.referrer)

        if image and allowed_file(image.filename):
            filename1 = secure_filename(image.filename)
            time = datetime.now().strftime("%d%m%Y%S%f")
            time += filename1
            filename1=time
            image.save(os.path.join(app.config['UPLOAD_FOLDER'],filename1)) 
        
        data = {
            'name': request.form['name'],
            'description': request.form['description'],
            'contact': request.form['contact'],
            'location': request.form['location'],
            'linki': request.form['linki'],
            'user_id': session['user_id'],
            'image' : filename1
        }
        Post.save(data)
        return redirect('/myconcern')
    return redirect('/')

@app.route('/edit/post/<int:id>')
def editPost(id):
    if 'user_id' in session:
        data = {
            'user_id': session['user_id'],
            'post_id': id
        }
        loggedUser = User.get_user_by_id(data)
        post = Post.get_post_by_id(data)
        print(loggedUser['id'])
        print(post['user_id'])
        if loggedUser['id'] == post['user_id']:
            return render_template('editPost.html', loggedUser = loggedUser, post= post)
        return redirect('/dashboard')
    return redirect('/')


@app.route('/post/<int:id>')
def viewPost(id):
    if 'user_id' in session:
        data = {
            'user_id': session['user_id'],
            'post_id': id
        }
        loggedUser = User.get_user_by_id(data)
        post = Post.get_post_by_id(data)
        savesNr = Post.get_post_savers(data)
        loggedUserSavedPost = User.get_user_saved_posts(data)


        return render_template('showOne.html',savedposts=loggedUserSavedPost, loggedUser = loggedUser, post= post, savesNr= savesNr)
    return redirect('/')

@app.route('/edit/post/<int:id>', methods = ['POST'])
def updatePost(id):
    if 'user_id' in session:
        data1 = {
            'user_id': session['user_id'],
            'post_id': id
        }
        loggedUser = User.get_user_by_id(data1)
        post = Post.get_post_by_id(data1)
        if loggedUser['id'] == post['user_id']:

            if not Post.validate_post(request.form):
                return redirect(request.referrer)
            
   
            if not request.files['image']:
                filename1 = post['image']
            image = request.files['image']
            
            if request.files['image']:
                if not allowed_file(image.filename):
                    flash('Image should be in png, jpg. jpeg format!', 'postImage')
                    return redirect(request.referrer)
        
               

                if image and allowed_file(image.filename):
                    filename1 = secure_filename(image.filename)
                    time = datetime.now().strftime("%d%m%Y%S%f")
                    time += filename1
                    filename1=time
                    image.save(os.path.join(app.config['UPLOAD_FOLDER'],filename1))
                    print(image)

            data = {
                'name': request.form['name'],
                'description': request.form['description'],
                'contact': request.form['contact'],
                'location': request.form['location'],
                'linki': request.form['linki'],
                'post_id': post['id'],
                'user_id': session['user_id'],
                'image' : filename1
            }
            Post.update(data)
            return redirect('/myconcern')
        return redirect('/concern')
    return redirect('/')

@app.route('/delete/post/<int:id>')
def deletePost(id):
    if 'user_id' in session:
        data = {
            'user_id': session['user_id'],
            'post_id': id
        }
        loggedUser = User.get_user_by_id(data)
        post = Post.get_post_by_id(data)
                
        if isinstance(post, bool):
            return "Error: Could not retrieve user or product data"
        Post.deleteAllSaves(data)
        Post.delete(data)
        return redirect(request.referrer)
    
    return redirect('/dashboard')
    return redirect('/')


@app.route('/save/<int:id>')
def savePost(id):
    if 'user_id' in session:
        data = {
            'user_id': session['user_id'],
            'post_id': id
        }
        
        savedPost = User.get_user_saved_posts(data)
        print("///////////////////////////////")
        print(savedPost)
        if id not in savedPost:
            Post.addSave(data)
            print("************************************")
            return redirect(request.referrer)
        return redirect(request.referrer)
    return redirect('/')


@app.route('/unsave/<int:id>')
def unsavePost(id):
    if 'user_id' in session:
        data = {
            'user_id': session['user_id'],
            'post_id': id
        }
        print("/------------------------------------")
        Post.unSave(data)
        return redirect(request.referrer)
    return redirect('/')

@app.route('/volunteer')
def volunteer():
    if 'user_id' not in session:
        return redirect('/')
    data = {
        'user_id': session['user_id']
    }

    loggedUser = User.get_user_by_id(data)
    posts = Post.get_all()
    loggedUserSavedPost = User.get_user_saved_posts(data)

    return render_template('volunteer.html', posts = posts, loggedUser=loggedUser, savedposts = loggedUserSavedPost)

@app.route('/process-message', methods=['POST'])
def process_message():
    data = request.get_json()
    message = data['message']

    # Call the OpenAI API to generate a response
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=message,
        max_tokens=50,
        temperature=0.7,
        n=1,
        stop=None
    )

    reply = response.choices[0].text.strip()

    return jsonify({'reply': reply})

@app.route('/thankyou')
def thankyou():
    if 'user_id' not in session:
        return redirect('/')
    data = {
        'user_id': session['user_id']
    }

    loggedUser = User.get_user_by_id(data)
    posts = Post.get_all()
    loggedUserSavedPost = User.get_user_saved_posts(data)

    return render_template('thankyou.html', posts = posts, loggedUser=loggedUser, savedposts = loggedUserSavedPost)
