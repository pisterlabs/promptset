from urllib import request
import openai
from flask import Flask, Response, request
import json
from flask_cors import CORS
import replicate



app = Flask(__name__)
CORS(app)

def getKey(service):
    with open('keys.json') as fid:
        data = json.load(fid)
        return data[service]
    



@app.route("/getRecipe", methods = ["GET"] )
def getRecipe():

    
    openai.api_key = getKey("openai")
    engines = openai.Engine.list()

    args = request.args.to_dict(flat=False)

    if('i' not in args.keys()):
        return 
    
    ingredients_list = args['i']

    
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt="Write a title and recipe based on these ingredients:\n" + "\n".join(ingredients_list),
        temperature=.95,
        max_tokens=240,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    # response.headers['Access-Control-Allow-Origin'] = '*'

    try:
        with open('response.json', "a") as fp:
            fp.write(json.dumps(response))
            fp.write("\n")
    except:
        print("Couldnt write to file")

    return response
    


@app.route("/getImage", methods=["GET"])
def getImage():

    rep = replicate.Client(api_token=getKey("replicate"))

    args = request.args.to_dict(flat=False)
    image_prompt = args['prompt'][0]
    print(image_prompt)

    model = rep.models.get("stability-ai/stable-diffusion")
    output = Response(model.predict(prompt= image_prompt))
    # output = "htoutps://replicate.delivery/pbxt/koGiXKGfIDTTPiZlyoojzE4iykjRPnAlhEXhghRWw4VleT1PA/out-0.png"
    # output.headers['Access-Control-Allow-Origin'] = '*'
    return output


if __name__ == '__main__':
    port = 8000 #the custom port you want
    app.run(host='0.0.0.0', port=port, debug=True)

@app.after_request
def add_header(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response