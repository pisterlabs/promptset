from flask import request,redirect,url_for,Flask
from dotenv import load_dotenv
from gpt import GPT
import os

# load the environment variables from the .env file
load_dotenv()
app = Flask(__name__)
gptAPI = GPT(os.environ.get('APIKEY'))

# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = b'_5#y2L"F4Q789789uioujkkljkl...8z\n\xec]/'

@app.route('/')
def index():
    ''' display a link to the general query page '''
    print('processing / route')
    return f'''
        <h1>GPT Demo</h1>
        <a href="{url_for('gptdemo')}">Ask questions to GPT</a>
        <hr>
        <h1>GPT Project Suggestions</h1>
        <a href="{url_for('gptProject')}">Get project suggestions</a>
        <hr>
        <h1>About</h1>
        <a href="{url_for('about')}">Learn about my project</a>
        <hr>
        <h1>Team</h1>
        <a href="{url_for('team')}">Meet the team</a>
        <hr>
        <h1>Form</h1>
        <a href="{url_for('form')}">Fill out a form</a>
        <hr>
    '''

@app.route('/about', methods=['GET', 'POST'])
def about():
    ''' display a link to the general query page '''
    print('processing /about route')
    return f'''
        <h1>About GPT Demo</h1>
        <p>This is a demo of the GPT API from OpenAI. It uses the Davinci model to answer questions.</p>
        <p>It is written in Python and uses the Flask framework to provide a web interface.</p>
        <p>It uses the dotenv library to read the API key from a .env file.</p>
        <p>It uses the openai library to make the API calls.</p>
        <p>It uses the requests library to make the API calls.</p>
        <a href="{url_for('index')}">Go back to the main page</a>
    '''

@app.route('/team', methods=['GET', 'POST'])
def team():
    ''' display a link to the general query page '''
    print('processing /about route')
    return f'''
        <h1>Team page</h1>
        <p> This is the team page for the GPT Demo. </p>
        <hr>
        <h2>Team Members</h2>
        <h3 style="text-align: center;"> Jamaine Genius </h3>
        <ul>

            <p>My name is Jamaine Genius
            <p> I am the sole member of this team
            <p> I am a student at Brandeis University 
            <p> I am a Computer Science and History double major
            <p> I am a rising senior and I created the GPT Project suggestions to help me create new Projects to work on
        </ul>
        <a href="{url_for('index')}">Go back to the main page</a>
    '''

@app.route('/form', methods=['GET', 'POST'])
def form():
    ''' handle a get request by sending a form '''
    if request.method == 'POST':
        prompt = request.form['prompt']
        if 'get better at' in prompt:
            response = gptAPI.getNewProject(prompt)  # Call the getNewProject method
        else:
            response = gptAPI.getResponse(prompt)  # Call the getResponse method  
        return f'''
        <form method="post">
            <textarea name="prompt"></textarea>
            <p><input type="submit" value="Get Response"></p>
        </form>
        <h2>GPT Response:</h2>
        <pre>{response}</pre>
        '''
    else:
        # Render the form if the request is GET
        return '''
        <h1>Form page</h1>
         <form method="post">
            <textarea name="prompt"></textarea>
            <p><input type="submit" value="Get Response"></p>
        </form>
        '''

@app.route('/gptProject', methods=['GET', 'POST'])
def gptProject():
    ''' handle a get request by sending a form 
        and a post request by returning the GPT response
    '''
    if request.method == 'POST':
        prompt = request.form['prompt']
        answer = gptAPI.getNewProject(prompt)
        return f'''
        <h1>GPT Project Suggestions</h1>
        <pre style="bgcolor:yellow">{prompt}</pre>
        <hr>
        Here is the answer in text mode:
        <div style="border:thin solid black">{answer}</div>
        Here is the answer in "pre" mode:
        <pre style="border:thin solid black">{answer}</pre>
        <a href={url_for('gptProject')}> make another query</a>
        '''
    else:
        return '''
        <h1>GPT Project Suggestions</h1>
        Enter a project you want to work on below
        <form method="post">
            <textarea name="prompt"></textarea>
            <p><input type=submit value="get response">
        </form>
        '''

@app.route('/gptdemo', methods=['GET', 'POST'])
def gptdemo():
    ''' handle a get request by sending a form 
        and a post request by returning the GPT response
    '''
    if request.method == 'POST':
        prompt = request.form['prompt']
        answer = gptAPI.getResponse(prompt)
        return f'''
        <h1>GPT Demo</h1>
        <pre style="bgcolor:yellow">{prompt}</pre>
        <hr>
        Here is the answer in text mode:
        <div style="border:thin solid black">{answer}</div>
        Here is the answer in "pre" mode:
        <pre style="border:thin solid black">{answer}</pre>
        <a href={url_for('gptdemo')}> make another query</a>
        '''
    else:
        return '''
        <h1>GPT Demo App</h1>
        Enter your query below
        <form method="post">
            <textarea name="prompt"></textarea>
            <p><input type=submit value="get response">
        </form>
        '''

if __name__=='__main__':
    # run the code on port 5001, MacOS uses port 5000 for its own service :(
    app.run(debug=True,port=5001)