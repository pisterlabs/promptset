# Partie 1
from flask import Flask
from flask import request
from flask import render_template,redirect,url_for,session
import openai
import os
import json
app = Flask(__name__)
os.chdir("tp3")

api_key = "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"
openai.api_key = api_key

app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = 'super_secret_key'

# Test if a file content is a json
def is_json(myjson):
  try:
    json.loads(myjson)
  except ValueError as e:
    return False
  return True

@app.route('/')
def root():
    with open('static/data/data.json') as f:
        data = json.load(f)
    fields=list(data[0].keys())
    # print(fields)

    return render_template("accueil.j2",fields=fields)

@app.route('/search', methods=["POST"])
def search():
    with open("static/data/data.json") as f:
        data=json.load(f)
    # print(data)
    searched=request.form.get("search")
    field=request.form.get("field")
    if field =="all":
        results = [item for item in data if any(searched.lower() in item[key].lower() for key in item.keys())]
    else:
        # Uniquement pour le champ concerné
        results = [item for item in data if searched.lower() in item[field].lower()]
    # Si pas de résultat, pas de tableau
    found=False
    fields=[]
    # Si on a bien des résultats on se prépare à afficher le tableau
    if len(results)>0:
        fields=list(results[0].keys())
        found=True
    content = render_template("search.j2",searched=searched,numbRes=len(results),field=field,fields=fields,results=results,found=found)
    return content
@app.route('/change', methods=["GET","POST"])
def change():
    # si method POST
    if request.method == 'POST':
        # A-t-on bien un fichier ?
        print(request.files)
        if not('file' in request.files):
            print("(DEBUG) NO FILE")
            return render_template('change.j2')
        # Récupère le fichier JSON
        file = request.files['file']
        # Est un json ?
        if not(is_json(file.stream.read())):
            print("(DEBUG) NOT JSON")
            return render_template('change.j2')
        # On remet la tete de lecture à 0
        file.stream.seek(0)
        # On enregistre
        file.save("static/data/data.json")
        return redirect(url_for('root'))
    # Affiche le formulaire d'upload
    return render_template('change.j2')

@app.route('/chat', methods=["POST","GET"])
def chat():
    if 'perso' in request.form: # test si provient de l'accueil
        session["perso"]=request.form.get("perso")
        return render_template("chat.j2",perso=session["perso"])
    elif 'question' in request.form: # test si provient du chat lui-même
        print("non fait")
            



if __name__ == '__main__':
    app.run(debug=True)
