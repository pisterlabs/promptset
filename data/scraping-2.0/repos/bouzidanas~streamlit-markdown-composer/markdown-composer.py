##################################  AI MARKDOWN WRITING ASSISTANT  ####################################
##                                                                                                   ##
## This is a Streamlit app that uses the OpenAI API to generate markdown text based on user input.   ##
##                                                                                                   ##
## AUTHOR: Anas Bouzid                                                                               ##
## DATE: 2023-06-13                                                                                  ##
##                                                                                                   ##
############################################  WARNING!  ###############################################
##                                                                                                   ##
## - This app is not affiliated with OpenAI.                                                         ##
##                                                                                                   ##
## - This app is a work in progress. It is not yet ready for production use. Use at your own risk.   ##
##                                                                                                   ##
## - This app requires OPENAI API access (key, org) and the author is not responsible or liable for  ##
##   any charges incurred by the user.                                                               ##
##                                                                                                   ##
##########################################  AUTHORS TODO  #############################################
##                                                                                                   ##
## - Add a feature that allows user to set `frequency_penalty` and `presence_penalty` api options.   ##
##   This could improve ai responses! See https://platform.openai.com/docs/api-reference/chat/create ##
##                                                                                                   ##
## - Instead of the current messy, manual formatting, use regex to mold AI responses into something  ##
##   that fits the context (remove repeated prefix ending, suffix beginning and query text in        ##
##   response for example).                                                                          ##
##                                                                                                   ##
## - Put logic that is duplicated in multiple functions in their own functions                       ##
##                                                                                                   ##
## - Persist state across sessions (e.g. settings, messages, etc.). Also, add save/download feature  ##
##                                                                                                   ##
## - Better logic for completing a sentence. If the cursor is in an incomplete sentence,             ##
##   the `completion_type` should be set to `sentence` and the ai prompt should be improved to get   ##
##   a more fitting result. Currently, in some situations, the app will try to rewrite the whole     ##
##   paragraph instead.                                                                              ##
##                                                                                                   ##
## - Add a feature that allows user to set settings inside AI ask command so user doesnt have to     ##
##   leave the text editor to change settings with mouse. Current conception is to use command       ##
##   style formatting (ex: --before|after|none|section, --temp=0.6).                                 ##
##                                                                                                   ##
## - Add a feature (`Coffee` button next to `Ask`?) that supercharges the AI by using a strategy     ##
##   (called "Expert Prompting") where the AI is first asked to boil down the request to main        ##
##   subject matter, then asked for top experts on subject matter, and finally told to respond to    ##
##   original request role playing as the one of the top experts.                                    ##
##                                                                                                   ##
## - Add option in settings to change `best_of` parameter of OpenAI chatCompletion                   ##
##                                                                                                   ##
#######################################################################################################
import sys
import openai
import tiktoken
import streamlit as st
from code_editor import code_editor

st.set_page_config(initial_sidebar_state="collapsed")

best_of = 1

if 'mycontents' not in st.session_state:
    st.session_state.mycontents = ""
if 'display' not in st.session_state:
    st.session_state.display = ""
if 'retry' not in st.session_state:
    st.session_state.retry = 0
if 'tokens' not in st.session_state:
    st.session_state.tokens = 0
if 'temp' not in st.session_state:
    st.session_state.temp = 0.6
if 'input' not in st.session_state:
    st.session_state.input = 1024
if 'long' not in st.session_state:
    st.session_state.long = 512
if 'med' not in st.session_state:
    st.session_state.med = 128
if 'short' not in st.session_state:
    st.session_state.short = 32
if 'response' not in st.session_state:
    st.session_state.response = ""
if 'lselect' not in st.session_state:
    st.session_state.lselect = "auto"
if 'iselect' not in st.session_state:
    st.session_state.iselect = "all"
if 'pselect' not in st.session_state:
    st.session_state.pselect = "all"
if 'mselect' not in st.session_state:
    st.session_state.mselect = "3.5 turbo"

# if 'delta' not in st.session_state:
#     st.session_state.delta = 0
# if 'lastdelta' not in st.session_state:
#     st.session_state.lastdelta = 0

custom_app_css = """
<style>
div.row-widget.stButton > button:first-child {
  width: 100%;
}
div.row-widget.stButton > button:first-child > div {
  margin: 2px;
}

</style>
"""
st.markdown(custom_app_css, unsafe_allow_html=True)

## OpenAI API Credentials setup ====================================================
with st.sidebar:
    st.markdown("## Configuration")
    with st.expander("OpenAI API Credentials"):
        default_openai_key = ""
        default_openai_org = ""
        if openai.api_key and openai.organization:
            default_openai_key = openai.api_key
            default_openai_org = openai.organization
        openai_key = st.text_input("Key", type="password", value=default_openai_key)
        openai_org = st.text_input("Organization", type="password", value=default_openai_org)
        if openai_org and openai_key:
            openai.organization = openai_org
            openai.api_key = openai_key
            try:
                openai.Model.list()
                st.success("Success!! API credentials registered.")
            except:
                st.error("Invalid API credentials!")
        

if openai_key == "" or openai_org == "":
    st.warning(":arrow_left: Please provide an OpenAI API key and organization in the sidebar.")

## Functions ========================================================================
token_count_at_start = 0
def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    """Returns the number of tokens in a text string.
    Args:
        string (str): The text string to be tokenized.
        encoding_name (str, optional): The name of the encoding to be used. Defaults to "cl100k_base".
    Returns:
        int: The number of tokens in the string.
    """

    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

messages=[
          {"role": "system", "content": "You are a helpful markdown writing assistant."}
         ]
def query_newer_api(message_text, temperature=0.7, max_tokens=512 , model="gpt-3.5-turbo-0613"):
    """Queries the OpenAI API with the given parameters and returns the result.
    Args:
        message_text (str): The text prompt to be sent to the API.
        temperature (float, optional): The temperature parameter for the API. Defaults to 0.7.
        max_tokens (int, optional): The max_tokens parameter for the API. Defaults to 512.
        model (str, optional): The model parameter for the API. Defaults to "gpt-3.5-turbo-0613".
    Returns:
        str: The response from the API.
    """
    sys.stdout.write("\n----> query_newer_api() \n")
    number_of_tokens = st.session_state.tokens + num_tokens_from_string(message_text)
    messages.append({"role": "user", "content": message_text})
    try:
        sys.stdout.write("Querying OpenAI API...")
        output = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    )
    except openai.error.Timeout as e:
        #Handle timeout error, e.g. retry or log
        st.error(f"ERROR! OpenAI API request timed out: {e}")
        pass
    except openai.error.APIError as e:
        #Handle API error, e.g. retry or log
        st.error(f"ERROR! OpenAI API returned an API Error: {e}")
        pass
    except openai.error.APIConnectionError as e:
        #Handle connection error, e.g. check network or log
        st.error(f"ERROR! OpenAI API request failed to connect: {e}")
        pass
    except openai.error.InvalidRequestError as e:
        #Handle invalid request error, e.g. validate parameters or log
        st.error(f"ERROR! OpenAI API request was invalid: {e}")
        pass
    except openai.error.AuthenticationError as e:
        #Handle authentication error, e.g. check credentials or log
        st.error(f"ERROR! OpenAI API request was not authorized: {e}")
        pass
    except openai.error.PermissionError as e:
        #Handle permission error, e.g. check scope or log
        st.error(f"ERROR! OpenAI API request was not permitted: {e}")
        pass
    except openai.error.RateLimitError as e:
        #Handle rate limit error, e.g. wait or log
        st.error(f"ERROR! OpenAI API request exceeded rate limit. Please wait a few seconds and try again.")
        pass
    except Exception as e:
        #Handle other exceptions, e.g. log
        st.error(f"ERROR! OpenAI API request failed: {e}")
        pass

    if output and output.choices[0].message.content:
        st.session_state.tokens = number_of_tokens + num_tokens_from_string(output.choices[0].message.content)
        return output.choices[0].message.content
    else:
        return "Error: No response from AI."
    
def query_completions_api(prefix, suffix, temperature=0.7, max_tokens=512):
    """Queries the OpenAI API with the given parameters and returns the result.
    Args:
        prefix (str): The text to be completed.
        suffix (str): The text that will go after the response. This is to provide additional context.
        temperature (float, optional): The temperature parameter for the API. Defaults to 0.7.
        max_tokens (int, optional): The max_tokens parameter for the API. Defaults to 512.
    Returns:
        str: The response from the API.
    """
    sys.stdout.write("\n----> query_completions_api()\n")
    number_of_tokens = st.session_state.tokens + num_tokens_from_string(prefix + suffix)
    try:
        sys.stdout.write("Querying OpenAI API...")
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prefix,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            best_of=best_of
        )
    except openai.error.Timeout as e:
        #Handle timeout error, e.g. retry or log
        st.error(f"ERROR! OpenAI API request timed out: {e}")
        pass
    except openai.error.APIError as e:
        #Handle API error, e.g. retry or log
        st.error(f"ERROR! OpenAI API returned an API Error: {e}")
        pass
    except openai.error.APIConnectionError as e:
        #Handle connection error, e.g. check network or log
        st.error(f"ERROR! OpenAI API request failed to connect: {e}")
        pass
    except openai.error.InvalidRequestError as e:
        #Handle invalid request error, e.g. validate parameters or log
        st.error(f"ERROR! OpenAI API request was invalid: {e}")
        pass
    except openai.error.AuthenticationError as e:
        #Handle authentication error, e.g. check credentials or log
        st.error(f"ERROR! OpenAI API request was not authorized: {e}")
        pass
    except openai.error.PermissionError as e:
        #Handle permission error, e.g. check scope or log
        st.error(f"ERROR! OpenAI API request was not permitted: {e}")
        pass
    except openai.error.RateLimitError as e:
        #Handle rate limit error, e.g. wait or log
        st.error(f"ERROR! OpenAI API request exceeded rate limit. Please wait a few seconds and try again.")
        pass
    except Exception as e:
        #Handle other exceptions, e.g. log
        st.error(f"ERROR! OpenAI API request failed: {e}")
        pass

    if response.choices[0].text:
        st.session_state.tokens = number_of_tokens + num_tokens_from_string(response.choices[0].text)
    
    return response.choices[0].text
    
# # To do: use AI to fix sentences ------------------------------------------------------------- 
# def ai_fix(content_to_fix, context, temperature=0.5, max_tokens=512 , model="gpt-3.5-turbo"):
#     fix_messages = [
#           {"role": "system", "content": "You are a helpful markdown editing and fixing assistant."}
#          ]
#     fix_messages.append({"role": "user", "content": r"Fix [" + content_to_fix + r"] so that it fits inside ```" + context + r"``` in the location marked by [HERE]"})
#     number_of_tokens = st.session_state.tokens + num_tokens_from_string(fix_messages[-1]["content"])
#     output = openai.ChatCompletion.create(
#                 model=model,
#                 messages=fix_messages,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 )

def get_prefix_suffix_for_completion(doc_string, cursor_position, prefix_stop=False):
    """Gets the prefix and suffix for a completion based on the cursor position in the document string
    Args:
        doc_string (str): The document string to split
        cursor_position (dict): The cursor position in the document string
        prefix_stop (bool): Whether to cut the prefix off at the start of the section containing the cursor position
    Returns:
        prefix (str): The prefix for the completion
        suffix (str): The suffix for the completion
        completion_type (str): The type of completion needed (all, section, paragraph, sentence)
    """
    sys.stdout.write("--> get_prefix_suffix_for_completion()\n")

    completion_type = "paragraph"
    # Get the line number of the cursor position
    line_number = cursor_position["row"]
    # Get the line of the cursor position
    doc_lines = doc_string.split("\n")
    line = doc_lines[line_number]
    # Get the prefix and suffix for completion
    line_prefix = line[:cursor_position['column']]
    if len(doc_lines[:line_number]) == 0:
        prefix = line_prefix
    else:
        prefix = "\n".join(doc_lines[:line_number]) + "\n" + line_prefix
    line_suffix = line[cursor_position['column']:]
    if len(doc_lines[line_number+1:]) == 0:
        suffix = line_suffix
    else:
        suffix = line_suffix +"\n" + "\n".join(doc_lines[line_number+1:])

    line_prefix_no_space = line_prefix.replace("\t", "").replace(" ", "")
    line_suffix_no_space = line_suffix.replace("\t", "").replace(" ", "")
    
    if line_prefix_no_space != "" and line_suffix_no_space != "" or line_prefix.startswith("#"):
        completion_type = "sentence"
    elif line_prefix_no_space != "" or line_suffix_no_space != "":
        completion_type = "paragraph"
    elif line_prefix_no_space == "" and line_suffix_no_space == "" and (line_number == 0 or doc_lines[line_number-1] == "") and (line_number + 2 > len(doc_lines) or doc_lines[line_number+1] == "" ):
        if len(prefix.split("#")) > 1:
            completion_type = "section"
        else:
            completion_type = "all"

    if prefix_stop and (completion_type == "section" or completion_type == "all"):
        if len(prefix.split("#")) > 1:
            temp_prefix = prefix.split("#").pop()
            if temp_prefix.replace(" ", "").replace("\n", "").replace("\t", "") != "":
                prefix = "#" + temp_prefix


    return prefix, suffix, completion_type

def get_prefix_suffix_within_limit(prefix, suffix, max_tokens_input):
    """Cuts the prefix and suffix off according to the input token limit given. Half the limit 
    is allocated to each. If the prefix or suffix is shorter than half the limit, 
    the unneeded tokens are added to the other half.
    Args:
        prefix (str): The entire prefix for the completion
        suffix (str): The entire suffix for the completion
        max_tokens_input (int): The maximum number of tokens allowed in the input
    Returns:
        prefix (str): The prefix (or subset) for the completion
        suffix (str): The suffix (or subset) for the completion
    """
    sys.stdout.write("----> get_prefix_suffix_within_limit()\n")
    prefix_surplus_tokens = 0
    suffix_surplus_tokens = 0

    prefix_tokens = num_tokens_from_string(prefix)
    suffix_tokens = num_tokens_from_string(suffix)

    if suffix_tokens < max_tokens_input/2:
        prefix_surplus_tokens = max_tokens_input/2 - suffix_tokens
    elif prefix_tokens < max_tokens_input/2:
        suffix_surplus_tokens = max_tokens_input/2 - prefix_tokens

    while prefix_tokens > max_tokens_input/2 + prefix_surplus_tokens:
        prefix = "\n".join(prefix.split("\n").pop(0))
        prefix_tokens = num_tokens_from_string(prefix)

    while suffix_tokens > max_tokens_input/2 + suffix_surplus_tokens:
        suffix = "\n".join(suffix.split("\n").pop())
        suffix_tokens = num_tokens_from_string(suffix)

    return prefix, suffix

def ask_ai(prefix, suffix, command, temperature=0.7, max_tokens=512):
    """Requests a context-based completion task to be executed by the AI
    Args:
        prefix (str): The prefix for the completion 
        suffix (str): The suffix for the completion
        command (str): The command to be executed by the AI
        temperature (float): The temperature for the completion
        max_tokens (int): The maximum number of tokens allowed for the completion
    Returns:
        response (str): The response from the AI
    """
    sys.stdout.write("\n--> ask_ai()\n")
    is_beginning = prefix.replace("\n","").replace("\t","").replace(" ","") == ""
    is_end = suffix.replace("\n","").replace("\t","").replace(" ","") == ""
    if is_beginning:
        message_text = command + " in markdown format that goes at the beginning, before the following text:\n\n" + suffix
    elif is_end:
        message_text = command + " in markdown format that goes at the end, after the following text:\n\n" + prefix
    elif is_beginning and is_end:
        message_text = command
    else: 
        message_text = command + " in markdown format that goes in the location marked by [insert] that best fits with the surrounding context in the following text:" + prefix + "[insert]" + suffix
    
    response = query_newer_api(message_text=message_text, temperature=temperature, max_tokens=max_tokens)

    if "[insert]" in response:
        if prefix + "[insert]" in response:
            response = response.replace(response[:response.index(prefix + "[insert]")], "").replace(prefix + "[insert]", "")
        elif "[insert]" in response:
            response = response.replace(" [insert] ", " ").replace("[insert]", "")
        elif "\[insert\]" in response:
            response = response.replace(" \[insert\] ", " ").replace("\[insert\]", "")

        if prefix.endswith(" ") and response.startswith(" "):
            response = response[1:]

    if is_beginning:
        if response.split("\n")[0] in prefix:
            response_lines = response.split("\n")
            for i in range(len(response_lines)):
                if response_lines[i] not in prefix:
                    response = "\n".join(response_lines[i:])
                    break
    return response

def get_completion_for_type(prefix, suffix, completion_type, temperature, max_tokens_input, max_tokens_long, max_tokens_med, max_tokens_short, ai="3.5 turbo"):
    """Gets a completion for the given completion type
    Args:
        prefix (str): The prefix for the completion
        suffix (str): The suffix for the completion
        completion_type (str): The type of completion to be performed
        temperature (float): The temperature for the completion
        max_tokens_input (int): The maximum number of tokens allowed in the input
        max_tokens_long (int): The maximum number of tokens allowed for a long completion
        max_tokens_med (int): The maximum number of tokens allowed for a medium completion
        max_tokens_short (int): The maximum number of tokens allowed for a short completion
        ai (str): The AI to use for the completion
    Returns:
        response (str): The response from the AI
    """
    # Get the context before and after the insert location (cursor position)
    sys.stdout.write("\n--> get_completion_for_type()\n")
    prefix, suffix = get_prefix_suffix_within_limit(prefix, suffix, max_tokens_input)

    if completion_type == "section" or completion_type == "all":
        max_tokens = max_tokens_long
    elif completion_type == "paragraph":
        max_tokens = max_tokens_med
    elif completion_type == "sentence":
        max_tokens = max_tokens_short
    
    if ai == "3.5 turbo":
        if completion_type == "sentence":
            message_text = "Add text to the sentence in the location marked by [insert] that fits the context provided in the following markdown text:" + prefix + "[insert]" + suffix
            # message_text = 'Add text to the sentence in the location marked by [insert] that fits the context provided in the following markdown text.\n\n Text: """\n' + prefix + "[insert]" + suffix + '\n"""'
        elif completion_type == "paragraph":
            # message_text = "Insert new text in the location marked by [insert] that best fits with the paragraph in the following markdown text:" + prefix + "[insert]" + suffix
            message_text = "Insert new text in the location marked by [insert] that best fits with the paragraph in the following markdown text:" + prefix + "[insert]" + suffix
        elif completion_type == "section":
            message_text = "Insert new text in the location marked by [insert] that best fits with the context in the following markdown section:" + prefix + "[insert]" + suffix
            # message_text = 'Insert new text in the location marked by [insert] that best fits with the context in the following markdown section.\n\nText: """\n' + prefix + "[insert]" + suffix + '\n"""'
        else:
            # message_text = "Insert new text in the location marked by [insert] that best fits with the surrounding context in the following markdown text:" + prefix + "[insert]" + suffix
            message_text = "Insert new text in the location marked by [insert] that best fits with the surrounding context in the following markdown text:" + prefix + "[insert]" + suffix
        response = query_newer_api(message_text, temperature, max_tokens)
    else:
        response = query_completions_api(prefix, suffix, temperature, max_tokens)

    # Process the response, remove redundant text, and stop the completion where appropriate
    # This needs to be simplified and made more efficient and robust
    if prefix + "[insert]" in response:
        response = response.replace(response[:response.index(prefix + "[insert]")], "").replace(prefix + "[insert]", "")
    elif "[insert]" in response:
        response = response.replace(" [insert] ", " ").replace("[insert]", "")
    elif "\[insert\]" in response:
        response = response.replace(" \[insert\] ", " ").replace("\[insert\]", "")

    if prefix.endswith(" ") and response.startswith(" "):
        response = response[1:]

    if response.split("\n")[0] in prefix:
        response_lines = response.split("\n")
        for i in range(len(response_lines)):
            if response_lines[i] not in prefix:
                response = "\n".join(response_lines[i:])
                break

    if response != "" or response != "Error: No response from AI.":
        if completion_type == "paragraph":
            lines = response.split("\n\n")
            if len(lines) > 1:
                for i in range(len(lines)):
                    if lines[i].replace("\t", "").replace(" ", "") != "":
                        response = lines[i] + "\n\n"

            last_sent = prefix.split("\n").pop().strip()
            response = response.removeprefix(last_sent)
            first_sent = suffix.split("\n")[0].split(".")[0].split("!")[0].split("?")[0]
            response = response.removesuffix(first_sent.strip())
        elif completion_type == "sentence":
            response = response.split("\n")[0].split(".")
            if len(response) > 1:
                response = (response[0] + ".").split("!")
            else: 
                response = response[0].split("!")
            if len(response) > 1:
                response = (response[0] + "!").split("?")
            else: 
                response = response[0].split("?")
            if len(response) > 1:    
                response = response[0] + "?"
            else: 
                response = response[0]
            
            last_sent_w_punc = prefix.split("\n").pop().strip()
            last_sent = last_sent_w_punc.strip('.?!').split(".")[-1].split("!")[-1].split("?")[-1]
            response = response.removeprefix(last_sent)
            first_sent_w_punc = suffix.split("\n")[0]
            first_sent = first_sent_w_punc.split(".")[0].split("!")[0].split("?")[0]
            response = response.removesuffix(first_sent.strip()).removesuffix(first_sent_w_punc.strip())

            response_words = response.split(" ")
            response_words_stripped = []
            for word in response_words:
                response_words_stripped.append(word.strip(".,?!"))

            last_words = last_sent.split(" ")
            templast_words = last_words
            templast_words.reverse()
            last_word = ""
            for word in templast_words:
                if word.strip(".,?!") != "":
                    last_word = word.strip(".,?!")
                    break
            if last_word != "" and last_word in response_words:
                j = response_words.index(last_word)
                resp_start = " ".join(response_words[:j+1])
                if resp_start in last_sent:
                    response = response.replace(resp_start.strip(), "")
            first_words = first_sent.split(" ")
            first_word = ""
            for word in first_words:
                if word.strip(".,?!") != "":
                    first_word = word.strip(".,?!")
                    break
            if first_word != "" and first_word in response_words_stripped:
                j = response_words_stripped.index(first_word)
                resp_end = " ".join(response_words[j:])
                if resp_end in first_sent:
                    response = response.strip(".,?!").replace(resp_end.strip(".,?!"), "")

            temp_first_sent = first_sent.strip().strip(".,?!").strip()
            if temp_first_sent != "" and temp_first_sent in response:
                response = response[:response.index(temp_first_sent)]

            if last_sent.endswith(" "):
                response = response.removeprefix(" ").removeprefix(" ")
            if first_sent.startswith(" "):
                response = response.removesuffix(" ").removesuffix(" ")
        response = response.replace("\n[insert]\n", "").replace("\n[insert] \n", "").replace(" [insert] ", " ").replace("[insert]", "")
    return response

def get_completion(prefix, suffix, temperature, max_tokens_input, max_tokens_output, ai="3.5 turbo"):
    """Returns the completion.
    Args:
        prefix (str): The prefix of the completion.
        suffix (str): The suffix of the completion.
        temperature (float): The temperature of the completion.
        max_tokens_input (int): The maximum number of tokens in the context provided in the prompt.
        max_tokens_output (int): The maximum number of tokens in the output.
        ai (str): The AI to use. Defaults to "3.5 turbo".
    Returns:
        response (str): The completion.
    """
    # Get the completion
    sys.stdout.write("\n--> get_completion()\n")

    prefix, suffix = get_prefix_suffix_within_limit(prefix, suffix, max_tokens_input)

    if ai == "3.5 turbo":
        message_text = "Provide text that fits best in the location marked by [insert] in the following markdown text:" + prefix + "[insert]" + suffix
        response = query_newer_api(message_text, temperature, max_tokens_output)
    else:
        response = query_completions_api(prefix, suffix, temperature, max_tokens_output)

    return response

def get_commands(text):
    """Extracts the commands in the text, removes all commands from the text, and determines the position of the first command.
    Args:
        text (str): The text to get the commands from.
    Returns:
        final_text (str): The text without the commands.
        commands (list): The commands.
        cursor_pos (dict): The position of the first command.
    """
    sys.stdout.write("\n--> get_commands()\n")

    index = 0
    commands = []
    rest = []
    pieces = text.split("<!--")
    for part in pieces:
        if index == 0:
            rest.append(part)
        elif "-->" in part and index == 0:
            sys.stdout.write("error: missing <!--")
            return "error: missing <!--", commands, {"row": 0, "column": 0}
        elif len(part.split("-->")) > 2:
            sys.stdout.write("error: too many -->")
            return "error: missing <!--", commands, {"row": 0, "column": 0}
        elif "-->" in part:
            command = part.split("-->")[0].strip()
            if command != "":
                commands.append(command)
            remainder = part.split("-->")[1]
            rest.append("<!--~-->" + remainder)
        else:
            sys.stdout.write("error: missing -->")
            return "error: missing -->", commands, {"row": 0, "column": 0}
        index += 1
    
    new_text = "".join(rest)

    row = 0
    column = 0
    if "<!--~-->" in new_text:
        temp_new_text = new_text[:new_text.index("<!--~-->")]
        lines_before = temp_new_text.split("\n")
        row = len(lines_before) - 1
        column = len(lines_before[-1])

    final_text = new_text.replace("<!--~-->", "")

    cursor_pos = {"row": row, "column": column}

    return final_text, commands, cursor_pos


## App ======================================================

## Input to query AI directly. Possible future feature.
# ai_chat_input = st.text_input(
#         "Ask AI to write ...",
#         label_visibility="collapsed",
#         placeholder="Ask AI to write ...",
#     )

# Prepare setup settings and elements for editor
btn_settings_editor_btns = [{
                                "name": "Ink",
                                "feather": "PenTool",
                                "hasText": True,
                                "alwaysOn": True,
                                "commands": [["response", "ink"]],
                                "bindKey": {"win": "Alt-Shift-R", "mac": "Alt-Shift-R"},
                                "style": {"bottom": "-0.6rem", "right": "0.1rem"}
                            },
                            {
                                "name": "Ask",
                                "feather": "MessageSquare",
                                "hasText": True,
                                "alwaysOn": True,
                                "commands": [["insertstring", "<!-- Ask AI here -->"], "gotowordleft", "gotowordleft", "selectwordleft", "selectwordleft", "selectwordleft", "selectwordleft", "selectwordleft", "keepFocus"],
                                "style": {"bottom": "-0.6rem", "left": "4.7rem"},
                                "class": "flipped-order shifting"
                            },
                            {
                                "name": "Write",
                                "feather": "Edit",
                                "hasText": True,
                                "alwaysOn": True,
                                "commands": ["submit"],
                                "style": {"bottom": "-0.6rem", "left": "0.1rem"},
                                "class": "flipped-order"
                            }]

## Load CSS to apply to Code Editor from file (is passed to Code Editor)
with open('composer_code_editor_css.scss') as css_file:
    css_text = css_file.read()

## Construct component props dictionary (is passed to Code Editor)
comp_props = {
              "css": css_text, 
              "globalCSS": ":root {\n  --streamlit-dark-font-family: monospace;\n  ----streamlit-light-font-family: monospace;\n}\n\n#root button.flipped-order {\n  flex-direction: row-reverse;\n}"
            #   "globalCSS": ":root {\n  --streamlit-dark-font-family: monospace;\n  ----streamlit-light-font-family: monospace;\n}\n\nbody #root button.always-on > span {\n  margin-bottom: -1px;\n  opacity: 0;\n  transform: scale(0);\n  transition: opacity 300ms 150ms, transform 300ms 300ms;\n}\n\nbody:hover #root button.always-on > span {\n  margin-bottom: -1px;\n  opacity: 1;\n  transform: scale(1);\n}\n\n#root button.flipped-order {\n  flex-direction: row-reverse;\n}\n\nbody #root button.flipped-order > span {\n  width: 0;\n  transform-origin: left center;\n  transition: opacity 300ms 150ms, transform 300ms;\n}\n\nbody:hover #root button.flipped-order > span {\n  width: unset;\n}\n\n#root button.flipped-order.shifting {\n transition: transform 300ms;\n}\n\nbody:hover #root button.flipped-order.shifting {\n  transform: translate(50%);\n}\n"
             }

## Inner Ace editor props  (is passed to Code Editor)
props = {
         "enableBasicAutocompletion": False,
         "enableLiveAutocompletion": False,
         "enableSnippets": False,
         "scrollMargin": [18, 18, 0, 0],
         "placeholder": "Enter text and press `Ctrl-Enter` or Cmd-Enter` to use AI...",
        }

## Inner Ace editor additional options
options = {"wrap": 76}

## Code Editor component
response_dict = code_editor(st.session_state.mycontents, lang="markdown", height=1000, buttons=btn_settings_editor_btns, props=props, options=options, component_props=comp_props, allow_reset=True, key="code_editor")

# handle `Write` button click or Ctrl/Cmd-Enter keypress
if response_dict['type'] == "submit" and st.session_state.response == "":
    commands_list = []
    if "<!--" in response_dict['text'] and "-->" in response_dict['text']:
        text_minus_comments, commands_list, cursor_pos = get_commands(response_dict['text'])
        if commands_list == [] or type(commands_list) != list or type(commands_list[0]) != str:
            pref_final, suf_final, ctype = get_prefix_suffix_for_completion(response_dict['text'], response_dict['cursor'], prefix_stop= st.session_state.pselect != "all")
        elif commands_list != [] and type(commands_list) == list and type(commands_list[0]) == str:
            pref_final, suf_final, ctype = get_prefix_suffix_for_completion(text_minus_comments, cursor_pos, prefix_stop= st.session_state.pselect != "all")
    else:
        pref_final, suf_final, ctype = get_prefix_suffix_for_completion(response_dict['text'], response_dict['cursor'], prefix_stop= st.session_state.pselect != "all")
    
    
    if st.session_state.iselect == "none":
        pref, suf, ctype = "", "", "all"
    elif st.session_state.iselect == "before":
        pref, suf = pref_final, ""
    elif st.session_state.iselect == "after":
        pref, suf = "", suf_final
    else:
        pref, suf = pref_final, suf_final

    if commands_list != [] and type(commands_list) == list and type(commands_list[0]) == str:
        command = commands_list[0]
        if st.session_state.lselect == "auto" or st.session_state.lselect == "long":
            res = ask_ai(pref, suf, command, st.session_state.temp, st.session_state.long)
        elif st.session_state.lselect == "short":
            res = ask_ai(pref, suf, command, st.session_state.temp, st.session_state.short)
        elif st.session_state.lselect == "medium":
            res = ask_ai(pref, suf, command, st.session_state.temp, st.session_state.med)
    else:
        if st.session_state.lselect == "auto":
            res = get_completion_for_type(pref, suf, ctype, st.session_state.temp, st.session_state.input, st.session_state.long, st.session_state.med, st.session_state.short, ai=st.session_state.mselect)
        elif st.session_state.lselect == "short":
            res = get_completion(pref, suf, st.session_state.temp, st.session_state.input, st.session_state.short, ai=st.session_state.mselect)
        elif st.session_state.lselect == "medium":
            res = get_completion(pref, suf, st.session_state.temp, st.session_state.input, st.session_state.med, ai=st.session_state.mselect)
        elif st.session_state.lselect == "long":
            res = get_completion(pref, suf, st.session_state.temp, st.session_state.input, st.session_state.long, ai=st.session_state.mselect)

    if res:
        st.session_state.response = res
        st.session_state.retry = 0
    else:
        st.session_state.response = ""
        st.session_state.retry += 1

    st.session_state.mycontents = pref_final + st.session_state.response + suf_final
    if st.session_state.retry > 2:
        st.session_state.response = "Sorry, no completion found. Please try again."
        st.session_state.retry = 0
    st.experimental_rerun()

if response_dict['type'] == "submit" and st.session_state.response != "":
    st.session_state.response = ""

## Adds bar below code editor containing buttons and inputs to configure AI
col1, col2, col3, col4, col5, col6 = st.columns([1.4, 1.4, 1.2, 1.2, 1.2, 1.4])
st.session_state.mselect = col2.selectbox("Model", ["3.5 turbo", "davince 003"], label_visibility="collapsed", help="model to use")
st.session_state.lselect = col3.selectbox("Max tokens", ["auto", "short", "medium", "long"], label_visibility="collapsed", help="max token limit for request/completion result")
st.session_state.iselect = col4.selectbox("Context to include", ["all", "before", "after", "none"], label_visibility="collapsed", help="context to include")
st.session_state.pselect = col5.selectbox("Part of context", ["all", "section"], label_visibility="collapsed", help="part of context to include")
st.session_state.temp = col6.number_input("Temperature", min_value=0.0, max_value=2.0, value=st.session_state.temp, step=0.1, label_visibility="collapsed", help="temperature")
st.write("\n")
st.write("\n")

# Adds container containing app settings to sidebar
with st.sidebar:
    with st.expander("Settings"):
        st.markdown("#### Input")
        st.session_state.temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=st.session_state.temp, step=0.1)
        st.session_state.input = st.slider("Max tokens to use to provide context", min_value=64, max_value=2048, value=1024, step=16)
        st.markdown("#### Output")
        st.session_state.long = st.slider("Max tokens for section completion", min_value=64, max_value=2048, value=512, step=16)
        st.session_state.med = st.slider("Max tokens for paragraph completion", min_value=16, max_value=1024, value=128, step=8)
        st.session_state.short = st.slider("Max tokens for sentence completion", min_value=8, max_value=256, value=64, step=2)
        
if col1.button("redo") and st.session_state.mycontents != response_dict['text']:
    do_nothing = True

## Respond to Ink button click or `Alt+Shift+Enter` keypress
if response_dict['type'] == "ink" and response_dict['text'] != "" and st.session_state.display != response_dict['text']:
    st.session_state.display = response_dict['text']

## Display markdown output in expander below code editor
if st.session_state.display != "":
    with st.expander("Rendered result", expanded=True):
        st.markdown(st.session_state.display)

## Adds expander containing instructions and tips below code editor
with st.expander("Instructions & Tips"):
    st.write("### Instructions")
    st.write("#### Write: Have the AI write something for you")
    st.write("1. Place the cursor where you want the AI to write. If there is nothing in the editor, you need to provide some context (you could write a heading or paste some starting text). Alternatively, you can start with a command (Ask)")
    st.write("2. Click the `Write` button to tell AI to complete or add to your text. ")
    st.success("TIP: You can use spacing to your advantage here. For example, if you want the AI to write the next part of the section, press `Enter` twice after the heading (or the previous paragraph) and then do step \#2.")
    st.write("3. Click redo or change any setting to get AI to try again (replacing the previous result).")
    st.write("4. When you are happy with the result, click the `Ink` button to keep the changes. Whatever is inked will be rendered below the editor in the 'Rendered result' section.")
    st.info("NOTE: Once you click the `Ink` or `Write` button (or press the keyboard shortcuts), the `redo` button along with the setting inputs will no longer trigger a retry of the previous results.")
    st.success("TIP: If you have inked the response but want to retry the last AI result, you can press `Ctrl-Z` to undo, then place the cursor where you want the AI to try again and repeat step 2.")
    st.write("#### Ask: Ask AI to do something like provide a conclusion or reformat the text")
    st.write("1. Place the cursor where you want the AI to insert its response")
    st.write("2. Write a markdown comment (<!-- example request -->). You can also use the 'Ask' button to insert a comment you can edit.")
    st.write("3. When you are ready to process the request, click `Ctrl-Enter`")
    st.info("NOTE: The AI will not alter the text surrounding the comment. It will only insert its response in place of the comment.")
    st.info("NOTE: Only the first comment in the editor will be processed. Any subsequent comments will be removed during processing.")
    st.info("NOTE: gpt-3.5-turbo-0613 is used for requests. In other words, the model selection will be ignored.")
    st.success("TIP: A quick way to ask the AI without leaving the keyboard is to type the request on an empty line and the press `Ctrl-/`. You can also use vscode shortcuts to select the request and then press `Ctrl-/`. The latter might be more useful when you dont want the result inserted on an empty line.")
    st.write("#### Keyboard shortcuts:")
    st.write("1. `Ctrl-Enter` ---> Write")
    st.write("2. `Alt-Shift-Enter` ---> Ink")
    st.write("2. Select request + `Ctrl-/` ---> Ask")
    st.write("3. `Ctrl-Z` ---> Undo")
    st.write("4. `Ctrl-Y` ---> Redo")
    st.info("NOTE: You can use vscode shortcuts in the editor.")
