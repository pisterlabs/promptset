import os
from quart import Quart, render_template, websocket, request, jsonify
from quart_cors import cors
from openai import OpenAI

openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', default=None))

app = Quart(__name__)
app.session = None

app = cors(app, allow_origin=[os.environ.get("CORS_ALLOWED_ORIGINS")], allow_headers=['content-type','Authorization'],  max_age=86400, allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

app.config['DEBUG'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
app.config['SESSION_TYPE'] = 'redis'
app.config['TEST'] = True
app.config['QUART_CORS_ALLOW_HEADERS'] = "contenttype, Authorization"
# app.config['QUART_CORS_ALLOW_ORIGIN'] = os.environ.get("CORS_ALLOWED_ORIGINS")
# app.config['QUART_CORS_ALLOW_CREDENTIALS'] = True
# app.config['QUART_CORS_MAX_AGE'] = 86400
# app.config['QUART_CORS_ALLOW_METHODS'] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
app.config["SESSION_COOKIE_SAMESITE"] = None
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('SESSION_COOKIE_SECURE')  # Set to True if using HTTPS!
app.config["WEBSOCKET_MAX_SIZE"] = 1024 * 1024 * 100  # Maximum size set to 1MB (adjust as needed)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 100  # Setting the maximum request size to 100MB
 
