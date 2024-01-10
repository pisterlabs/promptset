from pathlib import Path
import time
import logging
import json
from utilities.agent_tools import *
from openai import OpenAI

oai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) 


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class Agent():
    def __init__(self, thread_id=None):
        self.client = oai
        self.assistant = self.client.beta.assistants.retrieve("asst_1mmBGElejOMUV71ScIeRRAZb")
        self.thread_id = thread_id
        self.run = None
        
    def create_message_and_get_response(self, content):
        '''create the message by adding it to an existing thread'''
        self.client.beta.threads.messages.create(
            thread_id=self.thread_id,
            role="user",
            content=content
        )

        # create a run of that thread
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=self.assistant.id
        )

        # Wait for the run to complete with exponential backoff and timeout
        run_status = run.status
        max_wait_time = 120  # Maximum wait time in seconds
        total_waited = 0
        wait_interval = 1  # Initial wait interval in seconds

        while run_status not in ["completed", "requires_action", "failed", "cancelled", "expired"]:
            if total_waited >= max_wait_time:
                logging.warning("Timeout reached while waiting for the run to complete.")
                return "Request timed out."

            time.sleep(wait_interval)
            total_waited += wait_interval
            wait_interval = min(wait_interval * 2, 10)  # Double the wait interval, up to 10 seconds

            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread_id,
                run_id=run.id
            )
            run_status = run.status

        # Check if run requires an action
        if run_status == "requires_action":
            required_actions = run.required_action.submit_tool_outputs.model_dump()
            logging.info("CALLLING REQUIRED ACIONS ======================== %s", required_actions["tool_calls"])
            tools_output = []
            for action in required_actions["tool_calls"]:
                func_name = action["function"]["name"]
                arguments = json.loads(action["function"]["arguments"])
                if func_name == "search":
                    try:
                        search_processor = SearchProcessor()
                        output = search_processor.run(arguments["query"])
                        tools_output.append({
                            "tool_call_id": action["id"],
                            "output": output
                        })
                    except Exception as e:
                        self.client.beta.threads.runs.cancel(run_id=(run.id), thread_id=(self.thread_id))
                        return f"Function {func_name} has failed due to {e}"
                elif func_name == "create_image":
                    try:
                        output = create_image(arguments["description"])
                        tools_output.append({
                            "tool_call_id": action["id"],
                            "output": output
                        })
                    except Exception as e:
                        self.client.beta.threads.runs.cancel(run_id=(run.id), thread_id=(self.thread_id))
                        return f"something went wrong while executing the create_image function\nError: {e}"
                elif func_name == "analyze_images_with_captions":
                    try:
                        output = analyze_images_with_captions(image_url=arguments["image_url"], caption=arguments["caption"])
                        tools_output.append({
                            "tool_call_id": action["id"],
                            "output": output
                        })
                    except Exception as e:
                        self.client.beta.threads.runs.cancel(run_id=(run.id), thread_id=(self.thread_id))
                        return f"something went wrong while executing the analyze_images_with_captions function\nError: {e}"
                else:
                    logging.info("+++++++++++++++++++++++ FUNCTION REQUIRED NOT FOUND! ++++++++++++++++++++++++")
                    return f"Function {func_name} not found! It's not available for this user."

            self.client.beta.threads.runs.submit_tool_outputs(
                thread_id=self.thread_id,
                run_id=run.id,
                tool_outputs=tools_output,
            )
            time.sleep(2)
            run = self.client.beta.threads.runs.retrieve(
                        thread_id=self.thread_id,
                        run_id=run.id
                    )
            run_status = run.status
            # Exponential back off while waiting for run to complete or fail or for the run to be cancelled when the tool fails to work
            while run_status not in ["completed", "requires_action", "failed", "cancelled", "expired"]:
                if total_waited >= max_wait_time:
                    logging.warning("Timeout reached while waiting for the run to complete.")
                    return "Request timed out."
                else:
                    time.sleep(wait_interval)
                    total_waited += wait_interval
                    wait_interval = min(wait_interval * 2, 10)  # Double the wait interval, up to 10 seconds

                    run = self.client.beta.threads.runs.retrieve(
                        thread_id=self.thread_id,
                        run_id=run.id
                    )
                    run_status = run.status
            
            messages = self.client.beta.threads.messages.list(thread_id=self.thread_id)
            logging.info("+++++++++++++++++++++++ MESSAGES ++++++++++++++++++++++++ %s", messages.data)
            assistant_messages = [msg for msg in messages.data if msg.role == "assistant"]
            response = assistant_messages[0].content[0].text.value
            return response
        else:
            messages = self.client.beta.threads.messages.list(thread_id=self.thread_id)
            logging.info("+++++++++++++++++++++++ MESSAGES ++++++++++++++++++++++++ %s", messages.data)
            assistant_messages = [msg for msg in messages.data if msg.role == "assistant"]
            response = assistant_messages[0].content[0].text.value
            return response
        
    def create_audio(self, script):
        '''takes in a script and returns an audio file path'''
        try:
            speech_file_path = Path(__file__).parent / "speech.aac"
            response = self.client.audio.speech.create(
            model="tts-1",
            voice="nova",
            response_format="aac",
            input=script
            )
            response.stream_to_file(speech_file_path)
            return speech_file_path
        except Exception as e:
            logging.error("ERROR OCCURED======================================================================%s", e)
            return e

class Rogue(Agent):
    def __init__(self):
        super().__init__()
        self.thread_id = 'thread_jumec4yKfkbUQGOaLYQ4DyK4'

    def create_message_and_get_response(self, content):
        '''create the message by adding it to an existing thread'''
        self.client.beta.threads.messages.create(
            thread_id=self.thread_id,
            role="user",
            content=content
        )

        # create a run of that thread
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=self.assistant.id
        )

        # Wait for the run to complete with exponential backoff and timeout
        run_status = run.status
        max_wait_time = 120  # Maximum wait time in seconds
        total_waited = 0
        wait_interval = 1  # Initial wait interval in seconds

        while run_status not in ["completed", "requires_action", "failed"]:
            if total_waited >= max_wait_time:
                logging.warning("Timeout reached while waiting for the run to complete.")
                return "Request timed out."

            time.sleep(wait_interval)
            total_waited += wait_interval
            wait_interval = min(wait_interval * 2, 10)  # Double the wait interval, up to 10 seconds

            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread_id,
                run_id=run.id
            )
            run_status = run.status

        # Check if run requires an action
        if run_status == "requires_action":
            required_actions = run.required_action.submit_tool_outputs.model_dump()
            logging.info("CALLLING REQUIRED ACIONS ======================== %s", required_actions["tool_calls"])
            tools_output = []
            for action in required_actions["tool_calls"]:
                func_name = action["function"]["name"]
                logging.info("+++++++++++++++++++++++ FUNCTION NAME ++++++++++++++++++++++++ %s", func_name)
                arguments = json.loads(action["function"]["arguments"])
                if func_name == "write_tweet":
                    musk = ChiefTwit()
                    try:
                        output = musk.write_tweet((arguments["text"]))
                        tools_output.append({
                            "tool_call_id": action["id"],
                            "output": output
                        })
                        break
                    except Exception as e:
                        self.client.beta.threads.runs.cancel(run_id="run_TCi7Umz483eMPFzCQRumgMuA", thread_id="thread_jumec4yKfkbUQGOaLYQ4DyK4")
                        return f"something went wrong while executing the Tweet function\nError: {e}"
                elif func_name == "search":
                    try:
                        search_processor = SearchProcessor()
                        output = search_processor.run(arguments["query"])
                        tools_output.append({
                            "tool_call_id": action["id"],
                            "output": output
                        })
                        break
                    except Exception as e:
                        self.client.beta.threads.runs.cancel(run_id="run_TCi7Umz483eMPFzCQRumgMuA", thread_id="thread_jumec4yKfkbUQGOaLYQ4DyK4")
                        return f"something went wrong while executing the Search function\nError: {e}"
                elif func_name == "create_image":
                    try:
                        output = create_image(arguments["description"])
                        tools_output.append({
                            "tool_call_id": action["id"],
                            "output": output
                        })
                        break
                    except Exception as e:
                        self.client.beta.threads.runs.cancel(run_id="run_TCi7Umz483eMPFzCQRumgMuA", thread_id="thread_jumec4yKfkbUQGOaLYQ4DyK4")
                        return f"something went wrong while executing the create_image function\nError: {e}"
                elif func_name == "analyze_images_with_captions":
                    try:
                        output = analyze_images_with_captions(image_url=arguments["image_url"], caption=arguments["caption"])
                        tools_output.append({
                            "tool_call_id": action["id"],
                            "output": output
                        })
                        break
                    except Exception as e:
                        self.client.beta.threads.runs.cancel(run_id="run_TCi7Umz483eMPFzCQRumgMuA", thread_id="thread_jumec4yKfkbUQGOaLYQ4DyK4")
                        return f"something went wrong while executing the analyze_images_with_captions function\nError: {e}" 
                else:
                    logging.info("+++++++++++++++++++++++ FUNCTION REQUIRED NOT FOUND! ++++++++++++++++++++++++")
                    break

            self.client.beta.threads.runs.submit_tool_outputs(
                thread_id=self.thread_id,
                run_id=run.id,
                tool_outputs=tools_output,
            )
            time.sleep(2)
            run = self.client.beta.threads.runs.retrieve(
                        thread_id=self.thread_id,
                        run_id=run.id
                    )
            run_status = run.status
            # Exponential back off while waiting for run to complete or fail
            while run_status not in ["completed", "requires_action", "failed", "cancelled", "expired"]:
                if total_waited >= max_wait_time:
                    logging.warning("Timeout reached while waiting for the run to complete.")
                    logging.info("+++++++++++++++++++++++ RUN STATUS ++++++++++++++++++++++++ %s", run_status)
                    self.client.beta.threads.runs.cancel(run_id=run.id, thread_id=self.thread_id)
                    return "Request timed out."
                else:
                    time.sleep(wait_interval)
                    total_waited += wait_interval
                    wait_interval = min(wait_interval * 2, 10)  # Double the wait interval, up to 10 seconds

                    run = self.client.beta.threads.runs.retrieve(
                        thread_id=self.thread_id,
                        run_id=run.id
                    )
                    run_status = run.status
                    
            
            messages = self.client.beta.threads.messages.list(thread_id=self.thread_id)
            assistant_messages = [msg for msg in messages.data if msg.role == "assistant"]
            response = assistant_messages[0].content[0].text.value
            return response
        else:
            messages = self.client.beta.threads.messages.list(thread_id=self.thread_id)
            assistant_messages = [msg for msg in messages.data if msg.role == "assistant"]
            response = assistant_messages[0].content[0].text.value
            return response
                 
class Kim(Agent):
    def __init__(self, thread_id):
        super().__init__()
        self.assistant = self.client.beta.assistants.retrieve("asst_P8iLGX94gzCwmRoB7HRQ7qNo")
        self.thread_id = thread_id

    def create_message_and_get_response(self, content):
        '''create the message by adding it to an existing thread'''
        self.client.beta.threads.messages.create(
            thread_id=self.thread_id,
            role="user",
            content=content
        )

        # create a run of that thread
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=self.assistant.id
        )

        # Wait for the run to complete with exponential backoff and timeout
        run_status = run.status
        max_wait_time = 120  # Maximum wait time in seconds
        total_waited = 0
        wait_interval = 1  # Initial wait interval in seconds

        while run_status not in ["completed", "requires_action", "failed", "cancelled", "expired"]:
            if total_waited >= max_wait_time:
                logging.warning("Timeout reached while waiting for the run to complete.")
                return "Request timed out."

            time.sleep(wait_interval)
            total_waited += wait_interval
            wait_interval = min(wait_interval * 2, 10)  # Double the wait interval, up to 10 seconds

            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread_id,
                run_id=run.id
            )
            run_status = run.status

        # Check if run requires an action
        if run_status == "requires_action":
            required_actions = run.required_action.submit_tool_outputs.model_dump()
            logging.info("CALLLING REQUIRED ACIONS ======================== %s", required_actions["tool_calls"])
            tools_output = []
            for action in required_actions["tool_calls"]:
                func_name = action["function"]["name"]
                arguments = json.loads(action["function"]["arguments"])
                if func_name == "search":
                    try:
                        search_processor = SearchProcessor()
                        output = search_processor.run(arguments["query"])
                        tools_output.append({
                            "tool_call_id": action["id"],
                            "output": output
                        })
                    except Exception as e:
                        self.client.beta.threads.runs.cancel(run_id=(run.id), thread_id=(self.thread_id))
                        return f"Function {func_name} has failed due to {e}"
                elif func_name == "create_image":
                    try:
                        output = create_image(arguments["description"])
                        tools_output.append({
                            "tool_call_id": action["id"],
                            "output": output
                        })
                    except Exception as e:
                        self.client.beta.threads.runs.cancel(run_id=(run.id), thread_id=(self.thread_id))
                        return f"something went wrong while executing the create_image function\nError: {e}"
                elif func_name == "analyze_images_with_captions":
                    try:
                        output = analyze_images_with_captions(image_url=arguments["image_url"], caption=arguments["caption"])
                        tools_output.append({
                            "tool_call_id": action["id"],
                            "output": output
                        })
                    except Exception as e:
                        self.client.beta.threads.runs.cancel(run_id=(run.id), thread_id=(self.thread_id))
                        return f"something went wrong while executing the analyze_images_with_captions function\nError: {e}"
                else:
                    logging.info("+++++++++++++++++++++++ FUNCTION REQUIRED NOT FOUND! ++++++++++++++++++++++++")

            self.client.beta.threads.runs.submit_tool_outputs(
                thread_id=self.thread_id,
                run_id=run.id,
                tool_outputs=tools_output,
            )
            time.sleep(2)
            run = self.client.beta.threads.runs.retrieve(
                        thread_id=self.thread_id,
                        run_id=run.id
                    )
            run_status = run.status
            # Exponential back off while waiting for run to complete or fail or for the run to be cancelled when the tool fails to work
            while run_status not in ["completed", "requires_action", "failed", "cancelled", "expired"]:
                if total_waited >= max_wait_time:
                    logging.warning("Timeout reached while waiting for the run to complete.")
                    return "Request timed out."
                else:
                    time.sleep(wait_interval)
                    total_waited += wait_interval
                    wait_interval = min(wait_interval * 2, 10)  # Double the wait interval, up to 10 seconds

                    run = self.client.beta.threads.runs.retrieve(
                        thread_id=self.thread_id,
                        run_id=run.id
                    )
                    run_status = run.status
            
            messages = self.client.beta.threads.messages.list(thread_id=self.thread_id)
            logging.info("+++++++++++++++++++++++ MESSAGES ++++++++++++++++++++++++ %s", messages.data)
            assistant_messages = [msg for msg in messages.data if msg.role == "assistant"]
            response = assistant_messages[0].content[0].text.value
            return response
        else:
            messages = self.client.beta.threads.messages.list(thread_id=self.thread_id)
            logging.info("+++++++++++++++++++++++ MESSAGES ++++++++++++++++++++++++ %s", messages.data)
            assistant_messages = [msg for msg in messages.data if msg.role == "assistant"]
            response = assistant_messages[0].content[0].text.value
            return response
