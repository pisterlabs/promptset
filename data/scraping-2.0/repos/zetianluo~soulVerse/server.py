from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from langchain_model.views import create_blueprint

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'secret!'
    CORS(app, resources={r"/*": {"origins": "*"}})
    socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins
    main = create_blueprint(socketio)  # Get the blueprint from views.py
    app.register_blueprint(main)

    @app.route("/")
    def index():
        """Route for the root URL"""
        return "Welcome to the Flask/Socket.IO server!"

    @app.route("/http-call")
    def http_call():
        """return JSON with string data as the value"""
        data = {'data':'This text was fetched using an HTTP call to server on render'}
        return jsonify(data)

    @socketio.on("connect")
    def connected():
        """event listener when client connects to the server"""
        print(request.sid)
        print("client has connected")
        emit("connect", {"data": f"id: {request.sid} is connected"})

    @socketio.on('data')
    def handle_message(data):
        """event listener when client types a message"""
        print("data from the front end: ",str(data))
        emit("data", {'data':data, 'id':request.sid}, broadcast=True)

    @socketio.on("disconnect")
    def disconnected():
        """event listener when client disconnects to the server"""
        print("user disconnected")
        emit("disconnect", f"user {request.sid} disconnected", broadcast=True)

    return app, socketio

if __name__ == '__main__':
    app, socketio = create_app()
    socketio.run(app, debug=True, port=5001)
