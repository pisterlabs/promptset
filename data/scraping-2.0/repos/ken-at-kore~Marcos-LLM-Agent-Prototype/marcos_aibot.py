import requests
import json
from typing import List
from typing import Tuple
import traceback
import streamlit as st
from datetime import datetime
import openai
import time



class StAiBot:
    """
    Streamlit AiBots orchestrate the various OpenAI GPT Assistant calls, execute function calls, and handle the StreamlitUI.
    """

    @staticmethod
    def initialize(streamlit_page_title:str="My AiBot", 
                   welcome_message:str=None,
                   assistant_functions:List[dict]=None
        ):
        print("\n\n\nAiBot: Initializing session.")
        bot = StAiBot(streamlit_page_title=streamlit_page_title, 
                             welcome_message=welcome_message, 
                             assistant_functions=assistant_functions
        )
        st.session_state['ai_bot'] = bot


    
    @staticmethod
    def is_initialized():
        return 'ai_bot' in st.session_state



    def __init__(self, **kwargs):
        """
        Initialize the AiBot. Retreive the OpenAI GPT Assistant object and create an Assistant Thread.
        """
        # Initialize fields
        self.streamlit_page_title = kwargs.get('streamlit_page_title')
        self.welcome_message = kwargs.get('welcome_message')

        # Initialize AiFunctions
        self.assistant_functions = kwargs.get('assistant_functions', {})

        # Get the OpenAI Assistant key (which specifies the GPT model)
        assistant_preference = st.secrets.get("OPENAI_ASSISTANT_PREFERENCE", "OPENAI_GPT_3_5_ASSISTANT_ID")
        assert assistant_preference in st.secrets, f"OpenAI Assistant ID {assistant_preference} not found in secrets.toml file."
        openai_assistant_id = st.secrets[assistant_preference]

        # Initialize OpenAI Assistant objects
        self.assistant = openai.beta.assistants.retrieve(openai_assistant_id)
        print(f"AiBot: Assistant data retrieved: {self.assistant}\n")
        self.bot_thread = openai.beta.threads.create()
        print(f"AiBot: Assistant Thread initialized: {self.bot_thread}")

        # Initialize internal configs
        self.do_cot = False
        self.max_function_errors_on_turn = 1
        self.max_main_gpt_calls_on_turn = 4

        # Streamlit wants to know the model though I don't think it uses it
        st.session_state["openai_model"] = self.assistant.model

        # Initialize the Streamlit page caption
        self.streamlit_page_caption = "Powered by Kore.ai."
        if 'gpt-3.5' in self.assistant.model:
            self.streamlit_page_caption += " (Model 3.5)"
        elif 'gpt-4' in self.assistant.model:
            self.streamlit_page_caption += " (Model 4)"



    @staticmethod
    def runBot():
        """
        Get the AiBot from the Streamlit session and run it.
        """
        bot = st.session_state['ai_bot']
        assert bot is not None, "StreamlitAiBot has not been initialized"
        bot.run()



    def run(self):
        """
        Run AiBot's main loop. The bot takes a turn.
        """
        print("AiBot: Running.")

        # Display title and caption
        # (This needs to happen on every Streamlit run)
        st.title(self.streamlit_page_title)
        if self.streamlit_page_caption is not None:
            st.caption(self.streamlit_page_caption)

        # Initialize UI messages
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": self.welcome_message}
            ]

        # Re-render UI messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Get, store and render user message
        if user_input := st.chat_input("Enter text here", key="real_input"):
            print(f"AiBot: User input: {user_input}")

            # Display user input
            st.session_state.messages.append({"role": "user", "content": user_input}) # Add it to the UI thread
            with st.chat_message("user"):
                st.markdown(user_input) # Render it

            # For personal purposes, let user print entire conversation to the logs
            if user_input == 'print convo':
                self.print_conversation()
                return

            # Add the user input to the OpenAI Assistant Thread
            print("AiBot: Adding user message.")
            thread_message = openai.beta.threads.messages.create(self.bot_thread.id, role="user", content=user_input)

            # Initialize processing counters
            self.function_error_count = 0
            self.call_and_process_count = 0

            # Call GPT with the input and process results
            self.call_and_process_gpt()

        else:
            print("AiBot: Didn't process input.")


    def call_and_process_gpt(self, looping_run_id=None, function_outputs=None):
        """
        Call the OpenAI GPT Assistant Run command to compute a chat completion then process the results.
        """
        # Keep track of how often call & processing happens this turn
        self.call_and_process_count += 1

        # Prepare assistant response UI
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            # Call OpenAI GPT (Response is streamed)
            gpt_call_count_this_round = 0
            bot_content_response = None
            required_func_calls = None
            while True:
                try:
                    gpt_call_count_this_round += 1
                    run_id, bot_content_response, required_func_calls = self.call_gpt(looping_run_id, function_outputs)
                    break

                # Handle errors / timeouts
                except Exception as e:
                    if gpt_call_count_this_round < 3:
                        print("AiBot: GPT call had an error. Trying again.")
                        traceback.print_exc()
                    else:
                        print("AiBot: GPT calls had too many errors. Displaying error message.")
                        error_message = "Sorry, there was an error. Please try again."
                        bot_content_response = error_message
                        break
                    
            there_are_required_func_calls = required_func_calls is not None and len(required_func_calls) > 0

            if not there_are_required_func_calls:
                message_placeholder.markdown(bot_content_response) # Display bot content
            else: 
                message_placeholder.markdown('Just a sec 🔍') # Display filler message while function executes

        # Handle no function call
        if not there_are_required_func_calls:
            print(f"AiBot: GPT assistant message: {bot_content_response}")
            st.session_state.messages.append({"role": "assistant", "content": bot_content_response}) # Display bot content on re-run

        # Handle function call & display message
        else:
            # Add bot content to UI messages
            if bot_content_response != "":
                st.session_state.messages.append({"role": "assistant", "content": bot_content_response})

            # Handle the function call
            self.handle_required_function_calls(required_func_calls, run_id)



    def call_gpt(self, looping_run_id=None, function_outputs=None) -> Tuple[str, str, List[dict], str]:
        """
        Call the OpenAI GPT Assistant Run command (either Create or Submit Tool Ouputs) and wait for completion.
        """
        bot_content_response = ""
        required_func_calls = []

        if function_outputs is None or function_outputs == []:
            print("AiBot: Creating Thread Run...")
            run_creation = openai.beta.threads.runs.create(thread_id=self.bot_thread.id, assistant_id=self.assistant.id)
            print("AiBot: Run created.")

        else:
            print("AiBot: Submitting function output...")
            assert looping_run_id is not None, "Error: Expected a looping_run_id in call_gpt"
            run_creation = openai.beta.threads.runs.submit_tool_outputs(
                thread_id=self.bot_thread.id, run_id=looping_run_id, tool_outputs=function_outputs
            )
            print("AiBot: Output submission Run created.")

        while True:
            time.sleep(0.2)
            run = openai.beta.threads.runs.retrieve(thread_id=self.bot_thread.id, run_id=run_creation.id)
            print("AiBot: Run status: " + run.status)
            if run.status not in ['queued', 'in_progress']:
                break

        print("AiBot: Run done. Status: " + run.status)

        if run.status == 'completed':
            messages = openai.beta.threads.messages.list(self.bot_thread.id)
            print(f"AiBot: Retrieving new assistant message text. Messages count: {len(messages.data)}")
            bot_content_response = messages.data[0].content[0].text.value

        elif run.status == 'requires_action':
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            print("AiBot: Got required tool calls: " + str(tool_calls))
            for tool_call in tool_calls:
                assert tool_call.type == "function", "Error: Expected run tool_call to be a function call"
                required_func_calls.append({'id': tool_call.id, 
                                            'function_name': tool_call.function.name, 
                                            'function_args': tool_call.function.arguments}
                )
        elif run.status == 'failed':
            last_error_message = run.last_error.message
            print(f"AiBot: Run failed. {last_error_message} \n{run}")
            bot_content_response = f"Error: {last_error_message}"

        else:
            print("AiBot: Got an unexpected Run status. Run: " + str(run))
            raise Exception("Error: Got an unexpected Run terminal status: " + run.status)

        return (run.id, bot_content_response, required_func_calls)
    


    def handle_required_function_calls(self, required_func_calls:List, run_id:str):
        """
        Handle required function calls by executing the function and then re-running the GPT.
        """
        print("AiBot: Handling required function calls")

        # Execute and process all required function calls
        function_outputs = []
        there_was_a_func_call_error = False
        for required_func_call in required_func_calls:

            # Execute the function call
            func_call_result = self.execute_function_call(required_func_call['function_name'], required_func_call['function_args'])
            print(f"AiBot: Function execution result: {func_call_result.value}")

            # Check for errors
            if func_call_result.is_an_error_result:
                there_was_a_func_call_error = True

            # Append results to the outputs
            function_outputs.append({'tool_call_id': required_func_call['id'], 'output': func_call_result.value})

        if there_was_a_func_call_error:
            self.function_error_count += 1

        # Recursively call this same function to process the function call results
        self.call_and_process_gpt(run_id, function_outputs)
    
    def execute_function_call(self, function_call_name:str, function_call_args:str) -> 'StAiBot.FunctionResult':
        """
        Execute the function call. Catch exceptions.
        """
        assert function_call_name in self.assistant_functions, f'Function {function_call_name} is not defined in the assistant function dictionary.'
        try:
            print("AiBot: Executing " + function_call_name)
            if function_call_args is None or function_call_args == '':
                function_call_args = "{}"
            func_call_result = self.assistant_functions[function_call_name](json.loads(function_call_args))
            assert isinstance(func_call_result, StAiBot.FunctionResult), f"func_call_results for {function_call_name} must be of type AiFunction.Result, not {type(func_call_result)}"
        except Exception as e:
            error_info = f"{e.__class__.__name__}: {str(e)}"
            print(f"AiBot: Error executing function {function_call_name}: '{error_info}'")
            traceback.print_exc()
            func_call_result = StAiBot.FunctionResult(f"Caught exception when executing function {function_call_name}: '{error_info}'", is_an_error_result=True)

        return func_call_result
    


    def print_conversation(self):
        """
        Helper design and development feature for printing the entire conversation to the logs.
        """
        messages = openai.beta.threads.messages.list(self.bot_thread.id)
        for message in reversed(messages.data):
            role = message.role.capitalize()
            content = message.content[0].text.value
            print(f"**{role}:** {content}\n")

        with st.chat_message("assistant"):
            st.markdown("Printed the conversation to the logs.")



    class FunctionResult:
        def __init__(self, value:str, is_an_error_result=False):
            self.value = value
            self.is_an_error_result = is_an_error_result

    
    






# ------------------------------------------------------------------ #
# ------------------------ MARCOS SPECIFIC CODE ------------------------ #

# MARCOS AI FUNCTIONS

def transfer_to_agent(args):
    return StAiBot.FunctionResult("Transferring the call to an agent")

def hangup_the_phone(args):
    return StAiBot.FunctionResult("The system will hang up the phone call after the assistant says goodbye.")

def place_order(args):
    tax_rate = 0.08  # Let's say the tax rate is 8%
    total_amount = 0

    for item in args.get('items', []):
        price_str = item.get('itemPrice', '0')

        # Remove any non-numeric characters except the decimal point
        clean_price_str = ''.join(c for c in price_str if c.isdigit() or c == '.')

        # Safely convert the price string to a float
        try:
            item_price = float(clean_price_str)
        except ValueError:
            return StAiBot.FunctionResult("Error parsing an item price", is_an_error_result=True)

        total_amount += item_price

    # Calculate tax and add it to the total amount
    tax_amount = total_amount * tax_rate
    total_amount += tax_amount

    return StAiBot.FunctionResult(f"Place order successful. Order total: ${total_amount}")


# RUN THE BOT

def run():

    # Initialize the bot (if not initialized)
    if not StAiBot.is_initialized():

        # Setup Streamlit page configs
        page_title = "Marco's"
        st.set_page_config(
            page_title=page_title,
            page_icon="🤖",
        )

        # Initialize the AIBot
        StAiBot.initialize(streamlit_page_title=page_title,
                                    welcome_message=open('prompts & content/welcome message.md').read(),
                                    assistant_functions={
                                        'transfer_to_agent': transfer_to_agent,
                                        'hangup_the_phone': hangup_the_phone,
                                        'place_order':place_order
                                    }
        )

    # Run the AIBot
    StAiBot.runBot()



if __name__ == "__main__":
    run()
