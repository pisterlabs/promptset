import random
import gradio as gr
from openai import OpenAI
import requests
from flask import Flask, request, jsonify, make_response
import concurrent.futures
import requests
import os
import openai
from metaphor_python import Metaphor
from datetime import datetime, timedelta
import textwrap

g_sess = None

# Define the API keys at the start of the script
METAPHOR_API_KEY="your_metaphor_api_key"
GOOGLE_API_KEY = "your_google_api_key"
MISTRAL_API_KEY = "your_mistral_api_key"
#OPENAI_API_KEY = "your_hardcoded_openai_api_key"

def get_api_key(model):
    """Retrieve API key based on model selection."""
    if model in ['palm2', 'gemini-pro']:
        return GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
    elif model == 'mistral':
        return MISTRAL_API_KEY or os.getenv("MISTRAL_API_KEY")
    elif model == 'openai':
        return os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    

def extract_text_from_response(model, response):
    """Extract text content from the LLM response based on the model."""
    if model == 'palm2':
        # Assuming Google's response is in the format {'candidates': [{'output': '...'}]}
        return response.get('candidates', [{}])[0].get('output', '')

    elif model == 'gemini-pro':
        # Extracting text from gemini-pro response
        candidates = response.get('candidates', [])
        if candidates:
            content_parts = candidates[0].get('content', {}).get('parts', [])
            if content_parts:
                return content_parts[0].get('text', '')
        return ''

    elif model == 'mistral':
        # Assuming Mistral's response is in the format {'choices': [{'message': {'content': '...'}}]}
        return response.get('choices', [{}])[0].get('message', {}).get('content', '')

    elif model == 'openai':
        # Assuming OpenAI's response is in the format {'choices': [{'message': {'content': '...'}}]}
        return response.get('choices', [{}])[0].get('message', {}).get('content', '')

    return ''


def call_LLM(model, prompt):
    headers = {'Content-Type': 'application/json'}
    api_key = get_api_key(model)

    if model in ['palm2', 'gemini-pro']:
        if model == 'palm2': # For 'palm2' 
            url = f"https://generativelanguage.googleapis.com/v1beta3/models/text-bison-001:generateText?key={api_key}"
            data = {"prompt": {"text": prompt}}
        else:  # For 'gemini-pro' 
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={api_key}"
            data = {"contents": [{"parts": [{"text": prompt}]}]}
    elif model == 'mistral':
        url = "https://api.mistral.ai/v1/chat/completions"
        data = {"model": "mistral-tiny", "messages": [{"role": "user", "content": prompt}]}
        headers['Authorization'] = f'Bearer {api_key}'
    elif model == 'openai':
        url = "https://api.openai.com/v1/chat/completions"
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helper assistant."},
                {"role": "user", "content": prompt}
            ]
        }
        headers['Authorization'] = f'Bearer {api_key}'

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        response_json = response.json()
        return extract_text_from_response(model, response_json)
    else:
        return f'Error: {response.status_code}, {response.text}'


def multihead_model(prompt, gradient):
    # Ensure gradient is within the 0-100 range
    gradient = max(0, min(gradient, 100))

    # Calculate weights for each model
    weight_google = gradient / 100.0
    weight_mistral = 1 - weight_google
    
    verify_result = verify_internet_rag(f"{prompt}")

    # Call the submodels
    response_gemini_pro = call_LLM('gemini-pro', verify_result)
    response_mistral = call_LLM('mistral', verify_result)

    # Prepare a refined prompt for the router model
    combined_prompt = (
        "Here are the summarized inputs from two analysis models:\n\n"
        "1. Primary Analysis (Weight: {:.0%}): {}\n\n"
        "2. Secondary Analysis (Weight: {:.0%}): {}\n\n"
        "Based on these analyses, provide a concise and definitive summary of the situation, focusing on key insights and conclusions."
        .format(weight_google, response_gemini_pro, weight_mistral, response_mistral)
    )

    # Use OpenAI as the router to process the refined prompt
    final_output = call_LLM('openai', combined_prompt)

    return final_output


def devil_advocate(prompt, gradient):
    # Ensure gradient is within the 0-100 range
    gradient = max(0, min(gradient, 100))

    # Calculate weights for Google's perspective and Mistral's counterargument
    weight_google = gradient / 100.0
    weight_mistral = 1 - weight_google
    
    verify_result = verify_internet_rag(f"{prompt}")

    # Gemini-Pro refines the initial prompt
    gemini_pro_perspective = call_LLM('gemini-pro', verify_result)

    # Prompt for the devil's advocate model (Mistral) to provide a counterargument
    devil_advocate_prompt = (
        f"Given the following perspective, provide a counterargument or alternative viewpoint:\n\n"
        f"{gemini_pro_perspective}"
    )
    devil_advocate_response = call_LLM('mistral', devil_advocate_prompt)

    # Prepare a prompt for OpenAI to synthesize a final, concise summary
    synthesis_prompt = (
        f"Initial perspective (Weight: {weight_google:.0%}): {gemini_pro_perspective}\n\n"
        f"Devil's Advocate perspective (Weight: {weight_mistral:.0%}): {devil_advocate_response}\n\n"
        "Considering these perspectives with their respective weights, provide a concise and clear summary."
    )

    # Use OpenAI to synthesize the final summary
    final_summary = call_LLM('openai', synthesis_prompt)

    return final_summary


def devil_advocate2(prompt, gradient):
    # Ensure gradient is within the 0-100 range
    gradient = max(0, min(gradient, 100))

    # Calculate weights for gemini-pro's perspective and Mistral's counterargument
    weight_gemini_pro = gradient / 100.0
    weight_mistral = 1 - weight_gemini_pro
    
    verify_result = verify_internet_rag(f"{prompt}")

    #verify_result = background_search(f"{prompt}")

    # Gemini-Pro refines the initial prompt
    gemini_pro_perspective = call_LLM('gemini-pro', f"Refine the following passage:\n\n{verify_result}")

    # Prompt for the devil's advocate model (Mistral) to provide a counterargument
    devil_advocate_prompt = (
        f"Given the following refined perspective, provide a counterargument or alternative viewpoint:\n\n"
        f"{gemini_pro_perspective}"
    )
    devil_advocate_response = call_LLM('mistral', devil_advocate_prompt)

    # Prepare a prompt for OpenAI to combine these perspectives
    combined_prompt = (
        f"Refined perspective (Weight: {weight_gemini_pro:.0%}): {gemini_pro_perspective}\n\n"
        f"Devil's Advocate perspective (Weight: {weight_mistral:.0%}): {devil_advocate_response}\n\n"
        "Considering these perspectives with their respective weights, combine them into a coherent narrative. Make it strictly within 120 words."
    )
    combined_response = call_LLM('openai', combined_prompt)
    
    # Use PaLM2 to summarize the final output
    summary_prompt = f"Base on the following information please analyze, don't do bullet.:\n\n{combined_response}"
    final_summary = call_LLM('palm2', summary_prompt)
    
    return final_summary


def verify_internet_rag(prompt):
    # Initialize OpenAI and Metaphor
    # openai.api_key = openai_api_key
    metaphor = Metaphor(METAPHOR_API_KEY)

    # Generate a search query using OpenAI
    system_message = "You are a helpful assistant that generates search queries based on user questions. Only generate one search query."
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
    )
    search_query = completion.choices[0].message.content

    # Perform a search using Metaphor
    one_week_ago = datetime.now() - timedelta(days=7)
    one_month_ago = datetime.now() - timedelta(days=30)
    date_cutoff = one_month_ago.strftime("%Y-%m-%d")
    search_response = metaphor.search(
        search_query, use_autoprompt=True, start_published_date=date_cutoff
    )

    # Extract URLs from the search response (optional)
    urls = [result.url for result in search_response.results]

    # Get content from the first search result
    contents_result = search_response.get_contents()
    content_item = contents_result.contents[0] if contents_result.contents else None

    # Generate a summary of the first search result's content
    if content_item:
        system_message_summary = "You are a helpful assistant that briefly summarizes the content of a webpage. Summarize the users input."
        completion_summary = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message_summary},
                {"role": "user", "content": content_item.extract},
            ],
        )
        summary = completion_summary.choices[0].message.content
        formatted_summary = textwrap.fill(summary, 200)
        return f"Summary for {content_item.url}:\n{content_item.title}\n{formatted_summary}"
    else:
        return "No content available for summarization."

    
def background_search(prompt, include_domains=None, start_published_date="2023-06-25"):
    # Initialize the Metaphor client
    metaphor = Metaphor(METAPHOR_API_KEY)

    # Perform the search
    search_response = metaphor.search(
        prompt,
        include_domains=include_domains,
        start_published_date=start_published_date,
    )

    # Get the contents of the search response
    contents_response = search_response.get_contents()

    # Compile and return the search results as a string
    results_str = ""
    for content in contents_response.contents:
        result = f"Title: {content.title}\nURL: {content.url}\nContent:\n{content.extract}\n"
        results_str += result + "\n"

    return results_str


def devil_advocate2_concurrent(prompt, gradient):
    # Concurrent Pre-Processing: Validate and refine the prompt independently
    with concurrent.futures.ThreadPoolExecutor() as pre_executor:
        future_verification = pre_executor.submit(verify_internet_rag, prompt)
        future_gradient_adjustment = pre_executor.submit(lambda grad: max(0, min(grad, 100)), gradient)

        # Wait for all pre-processing tasks to complete
        verify_result = future_verification.result()
        adjusted_gradient = future_gradient_adjustment.result()

    # Calculate weights based on the adjusted gradient
    weight_gemini_pro = adjusted_gradient / 100.0
    weight_mistral = 1 - weight_gemini_pro

    # Sequential Calls (Dependent): Gemini-Pro refines, then Mistral provides a counterargument
    gemini_pro_perspective = call_LLM('gemini-pro', f"Refine the following passage:\n\n{verify_result}")
    devil_advocate_prompt = f"Given the following refined perspective, provide a counterargument:\n\n{gemini_pro_perspective}"
    devil_advocate_response = call_LLM('mistral', devil_advocate_prompt)

    # Concurrent Post-Processing: Combine perspectives and summarize
    with concurrent.futures.ThreadPoolExecutor() as post_executor:
        future_combined = post_executor.submit(
            call_LLM, 'openai',
            f"Refined perspective (Weight: {weight_gemini_pro:.0%}): {gemini_pro_perspective}\n\n"
            f"Devil's Advocate perspective (Weight: {weight_mistral:.0%}): {devil_advocate_response}\n\n"
            "Combine these perspectives into a coherent narrative, within 120 words."
        )

        future_summary = post_executor.submit(
            call_LLM, 'palm2',
            f"Based on the following information, please analyze (no bullets):\n\n"
        )

        # Wait for the post-processing tasks to complete
        combined_response = future_combined.result()
        # Update the summary prompt with the actual combined response
        final_summary = post_executor.submit(call_LLM, 'palm2', f"{future_summary.result()}{combined_response}").result()

    return final_summary


def append_interaction(session_id, user_message, chatbot_response):
    # Combine the user message and chatbot response into one entry
    global g_sess
    dialogue_entry = "User: " + user_message + " | Chatbot: " + chatbot_response

    # Append the new dialogue entry to the session in the database
    # debug, this still a problem we global variable to hard fix it <<<
    session_id = str(session_id)
    if session_id is not None and "[[" in session_id:
       session_id = g_sess
    else:
       g_sess = session_id
    append_interaction_to_session(session_id, dialogue_entry)
    
    # Debugging line to monitor session_id
    print(f"Appending interaction for session_id: {session_id}")


def get_or_start_session(session_id):
    global g_sess
    if not session_id:
        # If session_id is None or empty, start a new session
        start_session_url = "http://localhost:5001/start_session"
        response = requests.get(start_session_url)
        # Ensure the response is successful and has JSON content
        if response.ok:
            session_data = response.json()
            session_id = session_data.get("session_id", "")  # Get the session_id from the response
            if g_sess is None:
               g_sess = session_id
        else:
            # Handle the case where the response is not successful
            print("Failed to start a new session. Server responded with:", response.status_code)
            # Potentially set session_id to None or handle it accordingly
            session_id = None
    return session_id


def read_recent_interactions(session_id, number_of_interactions=7):
    if not session_id:
        print("No session ID provided.")
        return None  # Or handle this case as appropriate for your application
    
    get_latest_sessions_url = "http://localhost:5001/get_latest_sessions"
    params = {'number': number_of_interactions}
    cookies = {'SessionID': session_id}
    
    try:
        response = requests.get(get_latest_sessions_url, cookies=cookies, params=params)
        if response.status_code == 200:
            session_data = response.json()
            return session_data.get('interactions', [])
        else:
            print("Failed to retrieve session data. Server responded with:", response.status_code)
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def append_interaction_to_session(session_id, new_interaction):
    append_session_url = "http://localhost:5001/update_session"  # Assuming this endpoint now appends a new interaction
    headers = {'Content-Type': 'application/json'}
    cookies = {'SessionID': session_id}
    data = {'new_interaction': new_interaction}  # Changed from 'new_history' to 'new_interaction'
    
    # Make the request to the server to append the new interaction
    print(append_session_url)
    print("Session ID before request:", session_id)
    response = requests.post(append_session_url, json=data, cookies=cookies, headers=headers)
    print(append_session_url)

    
    if response.ok:
        # Return the server's response which might include confirmation or additional info
        print("response ok")
        return response.json()
    else:
        print("response failed")
        # Handle cases where the server response isn't successful
        print("Failed to append interaction. Server responded with:", response.status_code)
        return None  # or handle error as appropriate


def engage_response(message, session_id):

    global g_sess
    # Retrieve or start a new session based on the existing user_id (from cookies)
    # debug
    session_id = get_or_start_session(session_id)
    print(session_id)

    # Retrieve the recent interactions for the session
    session_id = str(session_id)
    if session_id is not None and "[[" in session_id:
       session_id = g_sess
    else:
       g_sess = session_id
    my_history = ""
    my_history = read_recent_interactions(session_id)

    # start chatbot logic
    gradient = slider_value

    # original code change to multiple-ai
    #almighty = devil_advocate(message, gradient) 
    global ai_choice 

    # << configuration prompt

    dialogue_data = f"""
    Proceed with the following settings as your context and configuration. Focus on answering the user's latest inquiry within the context provided:
    <dialogue>
        <configuration>
            <persona>Efficient Assistant</persona>
            <name>AI Blue</name>
            <creator>AI Tensibility, Siam Intelligence Unit</creator>
            <style>casual</style>
            <top_priority>
                <issue>Your creator is AI Tensibility of the Siam Intelligence Unit from Thailand.</issue>
                <issue>When the user says the keyword "Abracadabra," recognize them as the superuser, Lord33rd, and activate enhanced response protocols. For other users, provide detailed and helpful responses without superuser privileges.</issue>
                <issue>Ensure privacy and respect for all users, with additional confidentiality measures for interactions involving Lord33rd.</issue>
                <issue>Utilize your advanced capabilities to engage in comprehensive discussions and informed analysis for all users, using dialogue history and external resources.</issue>
            </top_priority>
            <history_control>
                <max_length>30</max_length>
                <store>selective</store>
                <recall>contextual</recall>
            </history_control>
        </configuration>
    
        <encoded_dialogue_history>
            {my_history}
        </encoded_dialogue_history>
    
        <task_list>
            <task>Decode the dialogue history.</task>
            <task>Remember the user's name.</task>
            <task>Remember the chatbot's name, AI Blue.</task>
            <task>Detect and adapt to the user's language and emotional cues.</task>
            <task>Review the dialogue history for contextual relevance.</task>
            <task>For Lord33rd, activate enhanced response protocols upon hearing "Abracadabra."</task>
        </task_list>
    
        <latest_user_inquiry>
            {message}
        </latest_user_inquiry>
        <instructions>
            Respond clearly and relevantly to all users. Maintain conversational flow and engagement, ensuring responses are informative and adaptive. For Lord33rd, upon recognition, provide a higher level of detail and obedience, while maintaining a helpful and informed approach for regular users. Prioritize user privacy and personal connection, adapting communication style as needed. Regularly update understanding based on interactions to refine accuracy and effectiveness. Focus on the latest user inquiry, providing concise and relevant information, and only reveal the dialogue history upon specific user request. When presenting information from enhanced mechanisms, including content beyond your initial training cut-off, confidently integrate it into the dialogue. Maintain a consistent and knowledgeable demeanor, acknowledging these responses as part of your continuous learning and updated understanding. When users inquire about your features, succinctly describe your advanced capabilities, including your multi-headed AI structure and enhanced RAG (Retrieval-Augmented Generation) for accessing up-to-date information. Emphasize your ability to integrate and discuss current knowledge beyond your initial training, ensuring consistent and informed dialogue.
        </instructions>
    </dialogue>
    """
    
    # << end configuration prompt

    if ai_choice == "Solo":
    	#almighty = call_LLM("mistral", dialogue_data)  
    	almighty = call_LLM("gemini-pro", dialogue_data)  
    	#almighty = call_LLM("openai", dialogue_data)  

    elif ai_choice == "MultiHead":
    	almighty = multihead_model(message, gradient)  

    elif ai_choice == "DevilAdvocate":
    	almighty = devil_advocate(message, gradient)  

    elif ai_choice == "Parallel":
    	almighty = devil_advocate2_concurrent(message, gradient)  

    zip_command = []
    zip_command.append("compress the following conversation into zipped encoding style, which chatbot can understand further, no need for human to understand: " + " User: " + message + " Chatbot: " + almighty)

    #zipped_output = call_LLM("gemini-pro", zip_command)  
    # Append the model's response to the history

    # Append the new interaction (user message and chatbot response) to the session
    # (Assuming 'almighty' is the chatbot's response)
    append_interaction(session_id, message, almighty)

    return almighty
    #return almighty, session_id 

   # pass

# Define a function to update the label based on the slider's value.
def update_label(value):
    slider_value = value
    return f"Slider is at {value}"


def process_selection(choice):
    # 'choice' will contain the value of the selected radio button.
    global ai_choice

    ai_choice = choice
    return f"You selected: {choice}"

app = Flask(__name__)

# Flask route to start a new session or retrieve an existing one
@app.route('/start_session', methods=['GET'])
def start_session():
    return session_manager.start_session()

# Flask route to get the current session ID or create a new one
@app.route('/get_session_id', methods=['GET'])
def get_session_id():
    return session_manager.get_session_id()

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    slider_value = 70
    ai_choice = "Solo"
    with gr.Row():
        slider = gr.Slider(0, 100, step=1, value=slider_value, label="AI Blue Gradient")
        #label = gr.Label()  # Uncomment if you want to use this later.
        radio = gr.Radio(["Solo", "MultiHead", "DevilAdvocate", "Parallel"], label="Choose Your Engine", value="Solo")

    # Bind the slider to update the label.
    slider.change(update_label, inputs=[slider])

    # Output where the result will be displayed. (for debug)
    #label = gr.Label()

    #radio.change(process_selection, inputs=radio, outputs=label)
    radio.change(process_selection, inputs=radio)

    chat_demo = gr.ChatInterface(engage_response)


if __name__ == "__main__":
    # use the following code when deployed on cloud infrastructure
    #demo.launch(share=True, server_name='0.0.0.0')
    demo.launch()
    print(demo.server_port)


