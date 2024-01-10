#! /usr/bin/env python3

import os
import threading
import http.server
import socketserver
import asyncio
import websockets
import json
import signal
import time
import sys

from models.model_factory import load_model
from models.OpenAIChatModel import OpenAIChatModel


import threading

ros_available = False
robot_available = False
robot_interface = None
try:
    import rospy
    ros_available = True
    rospy.init_node("ros_interface", anonymous=False)
except:
    print("Could not import rospy. Robot interface is not available.")
    ros_available = False

httpd = None
server_thread = None
model = None
asyncio_loop = None
ws_server = None
prompt_prefix = ""
prompt_suffix = ""


def serve_interface_html(args):
    global httpd

    class HTMLFileHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            with open(args.interface_page, "r") as file:
                html = file.read()
                html = html.replace(
                    "ws://localhost:8190", f"ws://{args.ip}:{args.ws_port}"
                )
            self.wfile.write(bytes(html, "utf8"))

    print(f"Starting server at http://{args.ip}:{args.port}")
    try:
        httpd = http.server.HTTPServer((args.ip, args.port), HTMLFileHandler)
        httpd.serve_forever()
    except Exception as e:
        print("HTTP server error: " + str(e))
        shutdown(None, None)


def generate_code(prompt):
    global model, prompt_prefix, prompt_suffix, code_timeout
    start_time = time.time()
    prompt = prompt_prefix + prompt + prompt_suffix
    stop_sequences = ["\n#", "\ndef ", "\nclass", "```"]
    code = model.generate_one(
        prompt=prompt,
        stop_sequences=stop_sequences,
        temperature=0.9,
        top_p=0.99999,
        max_tokens=512,
    )
    end_time = time.time()
    print(f"Code generation time: {round(end_time - start_time, 2)} seconds")
    if type(model) is not OpenAIChatModel:
        code = (prompt_suffix + code).strip()
    elif not code.startswith(prompt_suffix.strip()):
        code = (prompt_suffix + "\n" + code).strip()
    return code


def execute(code):
    global ros_available
    global robot_available
    global robot_interface
    if not ros_available:
        print("ROS not available. Ignoring execute request.")
    elif not robot_available:
        print("Robot not available. Ignoring execute request.")
    else:
        from robot_interface.src.robot_client_interface import execute_task_program

        robot_execution_thread = threading.Thread(
            target=execute_task_program,
            name="robot_execute",
            args=[code, robot_interface],
        )
        robot_execution_thread.start()


async def handle_message(websocket, message):
    data = json.loads(message)
    if data["type"] == "code":
        print("Received code generation request")
        code = generate_code(data["prompt"])
        response = {"code": f"{code}"}
        await websocket.send(json.dumps(response))
        if data["execute"]:
            print("Executing generated code")
            execute(code)
    elif data["type"] == "eval":
        print("Received eval request")
        # await eval(websocket, data)
    elif data["type"] == "execute":
        print("Executing generated code")
        execute(data["code"])
        await websocket.close()
    else:
        print("Unknown message type: " + data["type"])


async def ws_main(websocket, path):
    try:
        async for message in websocket:
            await handle_message(websocket, message)
    except websockets.exceptions.ConnectionClosed:
        pass


def start_completion_callback(args):
    global asyncio_loop, ws_server
    # Create an asyncio event loop
    asyncio_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(asyncio_loop)
    start_server = websockets.serve(ws_main, args.ip, args.ws_port)
    try:
        ws_server = asyncio_loop.run_until_complete(start_server)
        asyncio_loop.run_forever()
    except Exception as e:
        print("Websocket error: " + str(e))
        shutdown(None, None)


def shutdown(sig, frame):
    global ros_available, robot_available, robot_interface, server_thread, asyncio_loop, httpd, ws_server
    print(" Shutting down server.")
    if robot_available and ros_available and robot_interface is not None:
        robot_interface._cancel_goals()
        print("Waiting for 2s to preempt robot actions...")
        time.sleep(2)
    if ros_available:
        rospy.signal_shutdown("Shutting down Server")
    if httpd is not None:
        httpd.server_close()
        httpd.shutdown()
    if server_thread is not None and threading.current_thread() != server_thread:
        server_thread.join()
    if asyncio_loop is not None:
        for task in asyncio.all_tasks(loop=asyncio_loop):
            task.cancel()
        asyncio_loop.stop()
    if ws_server is not None:
        ws_server.close()
    if sig == signal.SIGINT or sig == signal.SIGTERM:
        exit_code = 0
    else:
        exit_code = 1
    sys.exit(exit_code)


def main():
    global server_thread
    global prompt_prefix
    global prompt_suffix
    global ros_available
    global robot_available
    global robot_interface
    global code_timeout
    global model
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()

    parser.add_argument("--ip", type=str, help="IP address", default="localhost")
    parser.add_argument(
        "--port", type=int, help="HTML server port number", default=8080
    )
    parser.add_argument(
        "--ws-port", type=int, help="Websocket server port number", default=8190
    )
    parser.add_argument(
        "--model-type",
        choices=["openai", "openai-chat", "palm", "automodel", "hf-textgen", "llama"],
        default="openai-chat",
    )
    parser.add_argument("--model-name", type=str, help="Model name", default="gpt-4")
    parser.add_argument(
        "--tgi-server-url",
        type=str,
        help="Text Generation Inference Client URL",
        default="http://127.0.0.1:8080",
    )
    parser.add_argument(
        "--prompt-prefix",
        type=Path,
        help="Prompt prefix",
        default="code_generation/prompt_prefix.py",
    )
    parser.add_argument(
        "--prompt-suffix",
        type=Path,
        help="Prompt suffix",
        default="code_generation/prompt_suffix.py",
    )
    parser.add_argument(
        "--interface-page",
        type=Path,
        help="Interface page",
        default="code_generation/interface.html",
    )
    parser.add_argument(
        "--max-workers", type=int, help="Maximum number of workers", default=1
    )
    parser.add_argument(
        "--robot",
        action="store_true",
        help="Flag to indicate if the robot is available",
    )
    parser.add_argument(
        "--timeout", type=int, help="Code generation timeout in seconds", default=20
    )

    if ros_available:
        args = parser.parse_args(rospy.myargv()[1:])
    else:
        args = parser.parse_args()

    robot_available = args.robot
    code_timeout = args.timeout

    signal.signal(signal.SIGINT, shutdown)

    if robot_available and ros_available:
        from robot_interface.src.robot_client_interface import RobotInterface

        robot_interface = RobotInterface()

    prompt_prefix = args.prompt_prefix.read_text()
    prompt_suffix = args.prompt_suffix.read_text()
    model = load_model(args)
    server_thread = threading.Thread(
        target=serve_interface_html, name="HTTP server thread", args=[args]
    )
    server_thread.start()

    start_completion_callback(args)


if __name__ == "__main__":
    main()
