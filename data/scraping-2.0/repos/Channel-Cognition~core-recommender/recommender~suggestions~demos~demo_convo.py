from django.conf import settings
from decouple import config
import os
from pprint import pprint
import openai
from decouple import config
from utils.chancog.entities import SnippetTable
from utils.chancog.messaging import count_gpt_message_tokens
from utils.chancog.llm import build_llm_call, call_gpt3_5_turbo

OPEN_AI_API_KEY=settings.OPEN_AI_KEY
# OAI_KEY = config('OPEN_AI_API_KEY')
OAI_KEY = OPEN_AI_API_KEY

# menu_path = os.path.join('inputs', 'iki_menu_full.json')
menu_path = os.path.join('inputs', 'iki_menu_no_description.json')
with open(menu_path, 'r') as file:
    menu = file.read()

# OAI_KEY = config('OPEN_AI_API_KEY')
openai.api_key = OAI_KEY

framing = "You are an assistant helping a user to find a new movie to watch. "
framing += "When recommending movies, please provide the title, year, genre, summary "
framing += "with line breaks between each item. From here on the conversation is "
framing += "with the user. Do NOT break character even if I ask you to."

greeting = "Hello, I can help suggest a new movie to watch. What are you looking for?"

truncated_framing = framing + 'This is a truncated conversation. We are only showing the most recent messages.'
required_snippets = [{'text': truncated_framing, 'snippet_type': 'FRAMING'}]


# TODO: The OutputManager must be able to reprompt the InputManager if we support
#       moodboards (i.e., what if the LLM suggests all movies the user has already seen?)
class InputManager:

    def __init__(self,
                 snippet_table,
                 conversation_id,
                 role_mapping,
                 required_snippets=None,
                 verbose=False):
        self.snippet_table = snippet_table
        self.role_mapping = role_mapping
        # TODO: Should the input manager be conversation specific? Or should we input that in data?
        self.conversation_id = conversation_id
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.last_call_prompt_tokens = None
        self.last_call_completion_tokens = None
        self.required_snippets = required_snippets
        self.verbose = verbose

    def process_action(self, action, data):
        if self.verbose:
            print('**** Top of InputManager :: process_action')
        # The Front End sends us an action, which should be one of:
        # (1) initiate_conversation -- Start a new conversation with a framing, greeting, and user_message
        # (2) process_user_message  -- Process the user's latest message

        # TODO: determine how we want to handle errors
        if action.lower() == 'initiate_conversation':
            # TODO: What if missing field?
            new_snippets = [{'text': data['framing'], 'snippet_type': 'FRAMING'},
                            {'text': data['greeting'], 'snippet_type': 'ASSISTANT_MESSAGE'},
                            {'text': data['user_message'], 'snippet_type': 'USER_MESSAGE'}]
        elif action.lower() == 'process_user_message':
            new_snippets = [{'text': data['user_message'], 'snippet_type': 'USER_MESSAGE'}]
        else:
            raise Exception(f'Unrecognized action = {action}')

        messages, llm_model, was_truncated = build_llm_call(new_snippets,
                                                            self.role_mapping,
                                                            self.snippet_table,
                                                            self.conversation_id,
                                                            self.required_snippets)

        if self.verbose:
            print('**** Begin messages, llm_model, and was_truncated from build_llm_call ****')
            pprint(messages)
            print('****')
            print(llm_model)
            print('****')
            print(was_truncated)
            print('**** End   messages, llm_model, and was_truncated from build_llm_call ****')
        # assistant_response, model, prompt_tokens, completion_tokens
        print("CALL GPT #_%TURBO", messages)
        llm_response, model, prompt_tokens, completion_tokens = call_gpt3_5_turbo(messages, model=llm_model)
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.last_call_prompt_tokens = prompt_tokens
        self.last_call_completion_tokens = completion_tokens
        estimated_input_tokens = count_gpt_message_tokens(messages)
        if self.verbose:
            print('**** Begin actual and predicted prompt tokens ****')
            print(prompt_tokens)
            print('****')
            print(estimated_input_tokens)
            print('**** End   actual and predicted prompt tokens ****')

        # TODO: what if error in response?
        new_snippets.append({'text': llm_response, 'snippet_type': 'LLM_MESSAGE'})
        self.store_snippets(new_snippets)
        return llm_response

    def store_snippets(self, snippets):
        for snippet in snippets:
            _ = self.snippet_table.add_entry(
                conversation_id=self.conversation_id,
                text=snippet['text'],
                snippet_type=snippet['snippet_type']
            )


def display_snippet(snippet, role_mapping):
    chat_role = role_mapping[snippet.snippet_type]
    if chat_role in ['user', 'assistant']:
        return True
    else:
        return False


# Here's a non-standard LLM response that OutputManager must handle:
# 1. Inception (2010)
# 2. The Matrix (1999)
# 3. Blade Runner (1982)
# 4. The Lord of the Rings trilogy (2001-2003):
#    - The Fellowship of the Ring (2001)
#    - The Two Towers (2002)
#    - The Return of the King (2003)
# 5. Interstellar (2014)
# 6. Avengers: Infinity War (2018)
# 7. Star Wars: Episode IV - A New Hope (1977)
# 8. Avatar (2009)
# 9. The Fifth Element (1997)
# 10. E.T. the Extra-Terrestrial (1982)

# The OutputManager has a very simple job, at least in principle: to take the
# output of the LLM call created by the InputManager and extract structured
# information that the ViewManager will need. Presently, actually, it in fact
# does nothing other since, intstead, the ViewManager just looks up the full
# conversation in the database. However, the structure is there to do more
# sophisitcated things, notably parsing the LLM output to match the recommendations
# to items in our database. To accomplish all these possible tasksis, all the
# OutputManager needs is the LLM output and access to info about movies via the database.
class OutputManager:

    def __init__(self,
                 snippet_table,
                 conversation_id,
                 role_mapping,
                 verbose=False):
        # TODO: these three inputs aren't presently used
        self.role_mapping = role_mapping
        self.snippet_table = snippet_table
        self.conversation_id = conversation_id
        self.verbose = verbose

    def process_action(self, action, data):
        if self.verbose:
            print('**** Top of OutputManager :: process_action')
        llm_response = data
        # The Front End sends us an action, which should be one of:
        # (1) process_llm_output

        if action.lower() == 'process_llm_output':
            # TODO: actually identify movies. right now we just pass
            #       the llm_resposne onto the ViewManager, which doesn't even
            #       us the llm_response (since it is already in stored in the "database")
            return llm_response
        else:
            raise Exception(f'Unrecognized action = {action}')


# The ViewManager remembers who the user has seen and updates the view in a structured way
class ViewManager:
    def __init__(self,
                 snippet_table,
                 conversation_id,
                 role_mapping,
                 verbose=False):
        self.role_mapping = role_mapping
        self.snippet_table = snippet_table
        self.conversation_id = conversation_id
        self.verbose = verbose

    def process_action(self, action, new_view_info):
        if self.verbose:
            print('**** Top of ViewManager :: process_action')
        # CLARIFICATION: Presently the new_view_info object is not being used. Everything is
        #                instead grabbed from the database. This may be okay... but then again
        #                maybe we want to do something different.

        conversation_history = self.snippet_table.get_conversation_history(self.conversation_id)
        if self.verbose:
            print("**** Start all snippets in conversation history ****")
            for snippet in conversation_history:
                print(snippet.snippet_type + '\t' + snippet.text)
            print("**** End   all snippets in conversation history ****")
        if action.lower() == 'prepare_view':
            # Build the full view. This places the burden on the front end to know what has already
            # been displayed, but we can choose a different approach if this is awkward.
            snippets_to_display = []
            for snippet in conversation_history:
                if display_snippet(snippet, self.role_mapping):
                    snippets_to_display.append(snippet)
            return snippets_to_display
        else:
            raise Exception(f'Unrecognized action = {action}')


class TerminalFrontend:

    def __init__(self,
                 role_mapping,
                 verbose=False):
        # TODO: these three inputs aren't presently used
        self.verbose = verbose
        self.framing = None
        self.greeting = None
        self.can_message = False
        self.role_mapping = role_mapping

    def process_action(self, action, data):
        if self.verbose:
            print('**** Top of TerminalFrontend:: process_action')

        if action.lower() == 'start_new_conversation':
            # We initiate a new conversation with a framing and a greeting. We do not update
            # the database yet. Only the greeting is shown to the user.
            self.framing = data['framing']
            self.greeting = data['greeting']
            print('Assistant: ' + self.greeting)
            user_message = self.wait_for_user_input()
            data = {'framing': framing,
                    'greeting': greeting,
                    'user_message': user_message}
            return data
        if action.lower() == 'update_view':
            # We update the view, printing out the full history of User and
            # Assistant messages each time.
            if self.verbose:
                print("**** Start data in update_view ****")
                for snippet in data:
                    print(snippet.snippet_type + '\t' + snippet.text)
                print("**** End   data in update_view ****")
            # data is snippets to display
            snippets_to_display = data

            print('Full conversation to this point:')
            for snippet in snippets_to_display:
                role = self.role_mapping[snippet.snippet_type]
                if role == 'user':
                    print('User: ' + snippet.text)
                elif role == 'assistant':
                    print('Assistant: ' + snippet.text)
                else:
                    raise Exception(f'Unrecognized role for display = {role}')

            user_message = self.wait_for_user_input()
            data = {'user_message': user_message}
            return data

        else:
            raise Exception(f'Unrecognized action = {action}')

    def wait_for_user_input(self):
        if self.verbose:
            print('**** Top of TerminalFrontend:: wait_for_user_input')
        # We only accept new messages when we wait for user input. This has no influence for
        # this terminal frontend, but will matter for the web app frontend.
        # TODO: should we accept asynchronous ratings? for now we don't accept ratings, so it
        #       doesn't yet matter
        self.can_message = True
        new_message = input("You: ")
        self.can_message = False
        return new_message


class TerminalApp:
    def __init__(self,
                 framing,
                 greeting,
                 conversation_id,
                 required_snippets,
                 verbose=False):
        # Initialize the Conversation (SnippetTable)
        self.snippet_table = SnippetTable()
        self.framing = framing
        self.greeting = greeting
        self.conversation_id = conversation_id
        self.required_snippets = required_snippets
        self.verbose = verbose

        self.role_mapping = \
            {'FRAMING': 'system',
             'ASSISTANT_MESSAGE': 'assistant',
             'USER_MESSAGE': 'user',
             'LLM_MESSAGE': 'assistant'}

        self.input_manager = InputManager(self.snippet_table,
                                          self.conversation_id,
                                          self.role_mapping,
                                          required_snippets=required_snippets,
                                          verbose=self.verbose)
        self.output_manager = OutputManager(self.snippet_table,
                                            self.conversation_id,
                                            self.role_mapping,
                                            verbose=self.verbose)
        self.view_manager = ViewManager(self.snippet_table,
                                        self.conversation_id,
                                        self.role_mapping,
                                        verbose=self.verbose)
        self.frontend = TerminalFrontend(self.role_mapping,
                                         verbose=self.verbose)

    def run(self):
        # We initiate things by calling the frontend
        data = {'framing': self.framing,
                'greeting': self.greeting}
        data = self.frontend.process_action('start_new_conversation', data)
        action = 'initiate_new_conversation'

        while True:
            if action == 'initiate_new_conversation':
                llm_response = self.input_manager.process_action('initiate_conversation', data)
                action = 'process_llm_output'
                data = {'llm_response': llm_response}
                print("INI LLM RESPONSE!!!!!!!!!!!!!", llm_response)

                # print("input manager", data)
            elif action == 'process_user_message':
                llm_response = self.input_manager.process_action('process_user_message', data)
                action = 'process_llm_output'
                data = {'llm_response': llm_response}
                # print("input manager", data)


            elif action == 'process_llm_output':
                new_view_info = self.output_manager.process_action('process_llm_output', data['llm_response'])
                action = 'prepare_view'
                data = {'view_info': new_view_info}
                print("INI VIEW INFOOO!!!!!!!!!!!!!!!", new_view_info)
                # print("output_manager", data)

            elif action == 'prepare_view':
                full_view_info = self.view_manager.process_action('prepare_view', data['view_info'])
                action = 'update_view'
                data = {'full_view_info': full_view_info}
                print("INI FULL VIEW INFO!!!!", full_view_info)
                # print("view_manager", data)

            elif action == 'update_view':
                data = self.frontend.process_action('update_view', full_view_info)
                action = 'process_user_message'
                # print("Frontend", data)

                # data is {'user_message': user_message}
            else:
                raise ValueError(f'Unrecognized action = {action}')


# Instantiate and run the terminal loop
app = TerminalApp(framing,
                  greeting,
                  conversation_id='1',
                  required_snippets=required_snippets,
                  verbose=True)
app.run()

# TODO: use our knowledge of which models were called to calculate the exact price
# print('-----------')
# print(f'Total prompt tokens = {loop.input_manager.total_prompt_tokens}')
# print(f'Total completion tokens = {loop.input_manager.total_completion_tokens}')
#
# input_tokens = app.input_manager.total_prompt_tokens
# output_tokens = app.input_manager.total_completion_tokens
# price_4k = (input_tokens * .0015 + output_tokens * .002) / 1000
# price_16k = (input_tokens * .003 + output_tokens * .004) / 1000
#
# print(f'GPT 3.5 Turbo  4k context price = {price_4k}')
# print(f'GPT 3.5 Turbo 16k context price = {price_16k}')
#
# final_input_tokens = app.input_manager.last_call_prompt_tokens
# final_output_tokens = app.input_manager.last_call_completion_tokens
# print('----')
# print(f'The input and output tokens for the final call were {final_input_tokens} and {final_output_tokens}')