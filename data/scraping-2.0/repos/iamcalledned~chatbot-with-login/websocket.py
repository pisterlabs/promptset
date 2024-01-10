import asyncio
import json
import logging
import ssl
from uuid import uuid4
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from starlette.endpoints import WebSocketEndpoint

from openai_utils_generate_answer import generate_answer
from config import Config
from chat_bot_database import create_db_pool, get_user_info_by_session_id, save_recipe_to_db, clear_user_session_id, get_user_id, favorite_recipe, get_recipe_for_printing
from process_recipe import process_recipe
from fastapi import APIRouter
from fastapi import Request

import redis
from redis.exceptions import RedisError
from get_recipe_card import get_recipe_card

import spacy
import re
from starlette.websockets import WebSocket

# Initialize Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Initialize FastAPI app
app = FastAPI()
router = APIRouter()

OPENAI_API_KEY = Config.OPENAI_API_KEY
connections = {}
tasks = {}  # Dictionary to track scheduled tasks for session cleanup

log_file_path = Config.LOG_PATH
LOG_FORMAT = 'WEBSOCKET - %(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s'

logging.basicConfig(
    filename=log_file_path,
    level=logging.DEBUG,
    format=LOG_FORMAT
)

# Async function to create a connection pool
async def create_pool():
    return await create_db_pool()


@router.post("/logout")
async def logout(request: Request):
    print("session_id passed in logout", session_id)
    session_id = request.json().get('session_id', '')
    if session_id:
        # Remove session data from Redis
        redis_client.delete(session_id)

        # Additional cleanup if necessary
        username = connections.pop(session_id, None)
        if username:
            print(f"User {username} logged out and disconnected.")

    return {"message": "Logged out successfully"}

@app.on_event("startup")
async def startup_event():
    app.state.pool = await create_pool()
    print("Database pool created")

# Function to schedule session data cleanup
async def clear_session_data_after_timeout(session_id, username):
    try:
        await asyncio.sleep(3600)  # wait an hour before cleaning
        # Check if the session still exists before clearing
        if redis_client.exists(session_id):
            redis_client.delete(session_id)
            await clear_user_session_id(app.state.pool, session_id)
            print(f"Session data cleared for user {username}")

            # Send a WebSocket message to the client to log out
            if username in connections:
                websocket = connections[username]
                await websocket.send_text(json.dumps({'action': 'force_logout'}))
    except Exception as e:
        print(f"Error in session cleanup task for {username}: {e}")

@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cookies = websocket.cookies
    session_id_from_cookies = cookies.get('session_id')
    print("session ID from cookies", session_id_from_cookies)
    username = None
    
    async def ping_client():
        while True:
            try:
                await websocket.send_text(json.dumps({'action': 'ping'}))
                await asyncio.sleep(30)  # Send a ping every 30 seconds
            except Exception as e:
                print(f"Error sending ping: {e}")
                break
    ping_task = asyncio.create_task(ping_client())


    try:
        initial_data = await websocket.receive_text()
        initial_data = json.loads(initial_data)
        #session_id = initial_data.get('session_id', '')
        session_id = session_id_from_cookies
        print("initial data", initial_data)
        print("got a live one, welcome")

        if session_id:
            session_data = redis_client.get(session_id)
            
            if session_data:
                session_data = json.loads(session_data)
                username = session_data['username']
                # Renew the session expiry time upon successful connection
                redis_client.expire(session_id, 3600)  # Reset expiry to another hour
            else:
                await websocket.send_text(json.dumps({'action': 'redirect_login', 'error': 'Invalid session'}))
                #await websocket.send_text(json.dumps({'error': 'Invalid session'}))
                return
        else:
            await websocket.send_text(json.dumps({'action': 'redirect_login', 'error': 'Session ID required'}))
            #await websocket.send_text(json.dumps({'error': 'Session ID required'}))
            return

        while True:
            data = await websocket.receive_text()
            data_dict = json.loads(data)
            message = data_dict.get('message', '')
            #session_id = data_dict.get('session_id', '')
            session_id = session_id_from_cookies
            
                # Validate session_id
            if not session_id or not redis_client.exists(session_id):
                await websocket.send_text(json.dumps({'action': 'redirect_login', 'error': 'Invalid or expired session'}))
                

            # Renew the session expiry time
            redis_client.expire(session_id, 3600)
            print("data dict", data_dict)
            print("data", data)
            #print("data_dict from receive_text:", data_dict)
            
            if data_dict.get('action') == 'pong':
                redis_client.expire(session_id, 3600)  # Reset expiry to another hour
                continue

            # Renew the session expiry time after receiving each message
            redis_client.expire(session_id, 3600)  # Reset expiry to another hour
            print("extended redis")
            
            if 'action' in data_dict and data_dict['action'] == 'save_recipe':
                # Handle the save recipe action
                
                # Initialize save_result with a default value
                save_result = 'not processed'  # You can set a default value that makes sense for your application  
                userID = await get_user_id(app.state.pool, username)
                recipe_id = data_dict.get('content')
                print("recipe _ID", recipe_id)
                save_result = await favorite_recipe(app.state.pool, userID, recipe_id)
                print("save result from websocket:", save_result)

                if save_result == 'Success':
                   save_result = 'success'
                   
                   print("save result:", save_result)
                await websocket.send_text(json.dumps({'action': 'recipe_saved', 'status': save_result}))
                continue
            
            if 'action' in data_dict and data_dict['action'] == 'print_recipe':
                # Handle the print_recipe recipe action
                
                # Initialize save_result with a default value
                
                recipe_id = data_dict.get('content')
                print("recipe _ID for printing", recipe_id)
                print_result = await get_recipe_for_printing(app.state.pool, recipe_id)
                print("print result from websocket:", print_result)

                await websocket.send_text(json.dumps({'action': 'recipe_printed', 'data': print_result}))
                continue
            else:
                # Handle regular messages
                message = data_dict.get('message', '')
                user_ip = "User IP"  # Placeholder for user IP
                uuid = str(uuid4())
                print("session ID", session_id)
                
                
                response_text, content_type, recipe_id = await generate_answer(app.state.pool, username, message, user_ip, uuid)
                response = {
                    'response': response_text,
                    'type': content_type,
                    'recipe_id': recipe_id
                }
                
                await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for user {username}")
        print(f"Connections: {connections}")
        print(f"sessionid:", session_id)

        # Attempt to clear user data from Redis
        if session_id:
            # Schedule the task instead of immediate deletion
            task = asyncio.create_task(clear_session_data_after_timeout(session_id, username))
            tasks[session_id] = task

        connections.pop(username, None)

    except Exception as e:
        print(f"Unhandled exception for user {username}: {e}")
        print("Exception Traceback: " + traceback.format_exc())
    finally:
        ping_task.cancel()

async def on_user_reconnect(username, session_id):
    if session_id in tasks:
        tasks[session_id].cancel()
        del tasks[session_id]
        print(f"Clear data task canceled for user {username}")


@router.post("/validate_session")
async def validate_session(request: Request):
    session_id = request.json().get('session_id', '')
    if session_id and redis_client.exists(session_id):
        return {"status": "valid"}
    else:
        return {"status": "invalid"}

# Run with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
