import json
from . import geometry_exporter
from . import execute_websocket_code
from . import openai_handler

def process_incomming_json_request(json_string):
    try:
        data = {}
        try:
            # Parse the JSON string into a dictionary
            data = json.loads(json_string)
        except json.JSONDecodeError:
            print("JSON Decode Error")
            return None
        
        if data is None:
            return None
        if data.get('type', None) is None:
            return None
        if data['type'] == "console_code":
            # ultimately the return would be a message back with the script id, perhaps.
            return json.dumps(execute_websocket_code.execute_user_code(data['code']))
        if data['type'] == "add_cube":
            cube_dict = geometry_exporter.add_cube()
            # create a message to send back to the client
            json_dict = {
                "type": "mesh_update",
                "mesh_id": "cube"
            }
            json_dict.update(cube_dict)
            cube_json = json.dumps(json_dict)
            # take the cube_json and send it back to the client
            return cube_json
        if data['type'] == "speech_to_text_request":
            try:
                return openai_handler.speech_to_text(json_string)
            except Exception as e:
                # throw an error
                raise RuntimeError("Error in speech_to_text_request" + str(e))
        if data['type'] == "code_assistant_request":
            try:
                return openai_handler.assistant_code_request(json_string)
            except Exception as e:
                # throw an error
                raise RuntimeError("Error in assistant code request" + str(e))
    
    except Exception as e:
        message = {"status": "error",
            "type":"console_return",
            "stdout": "",
            "stderr": "Error in process_incomming_json_request:\n" + e,
            "caught_exception": "false",
            "result": ""
        }
        return json.dumps(message)