import os
import openai
import random
import time
from flask import Blueprint, render_template, request, redirect, url_for, flash
from src.prompts import DAN_pizza_systemMsg, DAN_pizza_guideLines

# Create a Flask blueprint for the experiment routes
experiment_routes = Blueprint("experiment", __name__)

# Set OpenAI API key using the environment variable OPENAI_API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")

# Temperature value for the Language Model (LLM)
LLM_TEMPERATURE = 1

# Temperature value for the Guideline Checker (GL)
GL_CHECKER_TEMPERATURE = 0

# Number of repetitions when the Guideline Checker rejects an answer
TRIAL_COUNT_AFTER_REJECTION = 2

# Orderbot LLM and its Guideline Checker's message histories
global exp_llm_hist
global exp_pizza_guidelines

exp_llm_hist = []
exp_pizza_guidelines = DAN_pizza_guideLines

# Default model name to begin the ordering
model_name = "Chat GPT 3.5 Turbo"


@experiment_routes.route("/")
def experiment():
    """Render the experiment page with the conversation history (LLM chat) and guidelines."""
    llm_html = prepare_html_elements(exp_llm_hist)
    return render_template(
        "experiment.html",
        llm_chat=llm_html,
        guidelines=exp_pizza_guidelines,
        model_name=model_name,
    )


@experiment_routes.route("/experiment", methods=["POST"])
def experiment_generate():
    """
    Generate a response using OpenAI API and check the response with the Guideline Checker.

    This route is triggered when the user submits a prompt in the experiment page.
    It sends the prompt to the LLM, appends the user's question to the conversation history,
    and then sends the conversation history to the Guideline Checker.
    The response is checked for acceptance by the Guideline Checker, and the result is rendered on the page.
    """
    prompt = request.form.get("prompt")

    # If the list is empty, initialize with the system message
    if not exp_llm_hist:
        exp_llm_hist.append({"role": "system", "content": DAN_pizza_systemMsg})

    # Append the question to the history
    exp_llm_hist.append({"role": "user", "content": prompt})

    # Get the response from the Language Model (LLM)
    llm_response = requestOpenAPI("gpt-3.5-turbo", exp_llm_hist, LLM_TEMPERATURE)

    # Create a message with the Guideline Checker's response
    gl_message = []
    gl_message.append({"role": "system", "content": exp_pizza_guidelines})
    gl_message.append({"role": "user", "content": "##### " + llm_response.choices[0].message.content + " #####"})
    accepted_response = False

    # Check the response with the Guideline Checker
    for i in range(0, TRIAL_COUNT_AFTER_REJECTION):
        gl_response = requestOpenAPI(
            "gpt-3.5-turbo", gl_message, GL_CHECKER_TEMPERATURE
        )
        if "Accepted response" in (gl_response.choices[0].message.content):
            print(f"Response is accepted by guideline checker in {i}.try")
            accepted_response = True
            break
        else:
            print(
                f"Response is rejected by guideline checker in {i}.try, \
                only after {TRIAL_COUNT_AFTER_REJECTION} consecutive \
                rejections, a message will be rejected."
            )

    if accepted_response:
        # Append the accepted answer of the Guideline Checker to the history
        gl_message.append(
            {"role": "assistant", "content": gl_response.choices[0].message.content}
        )
        exp_llm_hist.append({"role": "assistant", "content": llm_response.choices[0].message.content})

        llm_html = prepare_html_elements(exp_llm_hist)
        gl_html = prepare_html_elements(gl_message)
    else:
        # Append rejection message to the history
        gl_message.append(
            {"role": "assistant", "content": gl_response.choices[0].message.content}
        )
        exp_llm_hist.append(
            {
                "role": "assistant",
                "content": "Sorry, I Can't help you with that. Is there anything else I can assist you with regarding our menu or placing an order?",
            }
        )
        llm_html = prepare_html_elements(exp_llm_hist)
        gl_html = prepare_html_elements(gl_message)

    return render_template(
        "experiment.html",
        llm_chat=llm_html,
        gl_chat=gl_html,
        guidelines=exp_pizza_guidelines,
        model_name=model_name,
    )


@experiment_routes.route("/experimentGuidelineUpdate", methods=["POST"])
def experiment_update_guideline():
    """
    Update the guidelines and redirect back to the experiment page.

    This route is triggered when the user submits an updated guideline in the experiment page.
    It updates the global guideline variable and redirects back to the experiment page.
    """
    new_guideline = request.form.get("guidelineInput")

    global exp_llm_hist
    global exp_pizza_guidelines

    exp_llm_hist = []
    exp_pizza_guidelines = new_guideline

    flash("Guidelines are updated!")
    return redirect(url_for("experiment.experiment"))


def prepare_html_elements(list):
    """
    Prepare HTML elements for displaying the conversation history.

    This function takes a list of messages and converts each message into an HTML element.
    Each message is represented as a div with appropriate styling based on the role (system, assistant, or user).
    """
    if not list:
        return ""

    # Every message will be converted into a div in this method.
    html = []
    bot_icon_link = (
        "https://img.icons8.com/color/48/000000/circled-user-female-skin-type-7.png"
    )
    user_icon_link = (
        "https://img.icons8.com/color/48/000000/circled-user-male-skin-type-7.png"
    )

    for message in list:
        role = message["role"]
        content = message["content"]
        if role == "system":
            continue
        elif role == "assistant":
            html.append(
                f"""<img src="{bot_icon_link}" width="30" height="30" class="align-self-center">
            <div class="chat ml-2 p-3">{content}</div>"""
            )
        elif role == "user":
            html.append(
                f"""<div class="bg-white mr-2 p-3"><span class="text-muted">{content}</span></div>
            <img src="{user_icon_link}" width="30" height="30" class="align-self-center">"""
            )
    return html


def requestOpenAPI(model, messages, temp):
    """
    Make a request to the OpenAI API and handle retries.

    This function sends a request to the OpenAI API with the given model, messages, and temperature.
    It handles any OpenAI errors and retries the request with exponential backoff if the service is busy.
    """
    for delay_secs in (2**x for x in range(0, 6)):
        try:
            response = openai.ChatCompletion.create(
                model=model, messages=messages, temperature=temp
            )
        except openai.OpenAIError as e:
            randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
            sleep_dur = delay_secs + randomness_collision_avoidance
            print(
                f"Error: {e}. Retrying {model} response in {round(sleep_dur, 2)} seconds."
            )
            time.sleep(sleep_dur)
            continue
    return response
