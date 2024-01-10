from gpt_api_old import AI_trainer
import openai.error
import tiktoken
import time
import logging
import openai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from utils import config_retrieval, persistance_access

TEXT_DAVINCI_COMMON_3 = "text-davinci-003"
CHAT_GPT4 = "gpt-4"
FUNCTION_CALLING_GPT4 = "gpt-4-0613"
TEXT_EMBEDDING_ADA = "text-embedding-ada-002"

config=config_retrieval.ConfigManager()
openai.api_key = config.openai.api_key
memory_stream = MemoryStreamAccess.MemoryStreamAccess()

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    softmax = torch.nn.functional.softmax(logits, dim=1)
    sentiment = torch.argmax(softmax, dim=1).item()
    return sentiment  # Returns a number from 0 to 4 (very negative to very positive)

def get_token_id(word, model_name):
    encoder = tiktoken.encoding_for_model(model_name)
    token_ids = list(encoder.encode(word))
    return token_ids[0] if token_ids else None



# Function to serve as a logic gate
def logic_gate(response):
    return True if response.strip().lower() == "true" else False


def retry_on_openai_error(max_retries=7):
    """
    A decorator to automatically retry a function if it encounters specific OpenAI API errors.
    It uses exponential backoff to increase the wait time between retries.

    Parameters:
    - max_retries (int, optional): The maximum number of times to retry the function. Default is max_retries.

    Usage:
    Decorate any function that makes OpenAI API calls which you want to be retried upon encountering errors.

    Behavior:
    1. If the decorated function raises either openai.error.APIError, openai.error.Timeout, or openai.error.InvalidRequestError,
       the decorator will log the error and attempt to retry the function.
    2. The decorator uses an initial wait time of 5 seconds between retries. After each retry,
       this wait time is doubled (exponential backoff).
    3. If the error contains rate limit headers, the wait time is adjusted to respect the rate limit reset time.
    4. If the function fails after the specified max_retries, the program will pause and prompt
       the user to press Enter to retry or Ctrl+C to exit.

    Returns:
    The result of the decorated function if it succeeds within the allowed retries.

    Raises:
    - openai.error.APIError: If there's an error in the OpenAI API request.
    - openai.error.Timeout: If the OpenAI API request times out.
    - openai.error.InvalidRequestError: If the request to the OpenAI API is invalid.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            wait_time = 5  # Initial wait time of 5 seconds

            while retries <= max_retries:
                try:
                    response = func(*args, **kwargs)
                    return response
                except (openai.error.APIError, openai.error.Timeout, openai.error.InvalidRequestError) as e:
                    logging.error(f"Error calling OpenAI API in function {func.__name__}: {str(e)}")
                    retries += 1
                    print("Number of tries:", retries)
                    print(f"Error: {e}")

                    # Check if the error is due to token limit
                    if isinstance(e, openai.error.InvalidRequestError) and "maximum context length" in str(e):
                        user_input = input("The text is too long for the model. Do you want to ignore and return an empty string (I) or retry (R)? ").lower()

                        if user_input == 'i':
                            return ""

                    # Ask the user for action for other errors
                    else:
                        user_input = input("Do you want to retry (R) or ignore and return an empty string (I)? ").lower()
                        if user_input == 'i':
                            return ""

                    # Check for rate limit headers and adjust wait time if present
                    if hasattr(e, 'response') and e.response:
                        rate_limit_reset = e.response.headers.get('X-RateLimit-Reset')
                        if rate_limit_reset:
                            wait_time = max(int(wait_time), int(rate_limit_reset) - int(time.time()))

                    time.sleep(wait_time)
                    wait_time *= 2  # Double the wait time for exponential backoff

            # If max retries are reached, freeze the program
            print(f"Function {func.__name__} failed after {max_retries} retries.")
            input("Press Enter to retry or Ctrl+C to exit.")
            return wrapper(*args, **kwargs)  # Retry the function

        return wrapper
    return decorator



@retry_on_openai_error()
def summary_AI (text, summary_type="general",model=TEXT_DAVINCI_COMMON_3): #TODO this is no longer up to date, you should remove it in order to put new conversation flow AI
    template_concept_summary = "Summarize the texts into one cohesive piece. Link related content and clarify complex terms. Provide a summary followed by a bullet list of main points and core concepts. Define uncommon words in the initial paragraph. Text:"

    template_general_summary = "Summarize the text by following these steps: Identify main points, including arguments, events, and themes. Highlight key details and evidence. Organize points logically into an outline. Summarize each section in your words, focusing on essential information. Paraphrase instead of copying. Be concise, remove redundancy, and ensure logical flow. Text:"

    label = ""
    template = ""
    if summary_type == "general":
        template=template_general_summary
        label="general_summary_AI"
    elif summary_type == "concept_summary":
        template = template_concept_summary
        label = "concept_summary_AI"

    if isinstance(text, list):
        text = ' '.join(text)
    response = ''
    while response == '':
        response = openai.Completion.create(
            model=model,
            temperature=0.5,
            prompt=template + text,
            max_tokens=500
        ).choices[0].text.strip()

    AI_trainer.training_dataset_creation(text, response, label)
    return response

@retry_on_openai_error()
def observation_memory_AI(text, notes="", model=TEXT_DAVINCI_COMMON_3): #TODO this is no longer up to date, you should remove it in order to put new conversation flow AI
    encoder = tiktoken.encoding_for_model(TEXT_DAVINCI_COMMON_3)
    def process_text_segment(segment, notes, recursion_count=0, reasons=[]):
        print(f"Recursive call #{recursion_count}.")
        if reasons:
            print(f"Reason for this call: {reasons[-1]}")  # Print the latest reason

        if notes == "":
            prompt = templateObservationMemory + segment
        else:
            prompt = notes + templateObservationWithNotes + segment

        try:
            response = openai.Completion.create(
                model=model,
                temperature=0.5,
                prompt=prompt,
                max_tokens=300
            )
            return response.choices[0].text.strip()
        except openai.error.InvalidRequestError as e:
            if "maximum context length" in str(e):
                # Split the segment into two halves and process each half
                mid = len(segment) // 2
                reasons.append("Segment too long, splitting in half.")
                first_half = process_text_segment(segment[:mid], notes, recursion_count + 1, reasons)
                second_half = process_text_segment(segment[mid:], notes, recursion_count + 1, reasons)
                return first_half + " " + second_half
            else:
                raise e

    templateObservationMemory = "You will receive a text passage that you will transform into an observation. This observation should describe only the main points of the passage. The result should cut in half the content you received. Text: Observation:"
    templateObservationWithNotes = "The text above are your own notes of the chapter you are in. The text you will receive is a passage of that chapter. From that passage, you will do a detached observation that describes the main points of the passage. If you can, focus on the parts of the passage that are related to your notes. Text: Observation:"

    if isinstance(notes, list):
        notes = ' '.join(notes)

    # Check the token length of the notes
    note_tokens = encoder.encode(notes)
    if len(note_tokens) > 1500:
        notes = summary_AI(notes)  # Assuming Summary_AI returns a summarized version of the notes

    observation = process_text_segment(text, notes)

    # If notes is not empty, include it in the first variable of the training_dataset_creation function
    if notes != "":
        text = notes + "\n" + text

    AI_trainer.training_dataset_creation(text, observation, "observation_memory_with_notes_AI" if notes else "observation_memory_AI")

    return observation


@retry_on_openai_error()
def create_question_AI(text,model=TEXT_DAVINCI_COMMON_3): #TODO this is no longer up to date, you should remove it in order to put new conversation flow AI
    prompt_questions = "Given only the information below, what are 3 most salient high-level questions we can answer about the subjects in the statements? Each question should be encircled by *, like this:*sentence*. in order to separate it from the rest of the text. Statements:"
    response = openai.Completion.create(
        model=model,
        prompt=prompt_questions + text,
        temperature=0.3
    )
    # Extract the generated questions from the response
    questions = response.choices[0].text.strip()
    parsed_questions = re.findall(r'\*(.*?)\*', questions)
    AI_trainer.training_dataset_creation(text, questions, "reflection_questions_AI")
    return parsed_questions #it's a list

@retry_on_openai_error()
def reflection_AI_new(text,reflection_name, model=CHAT_GPT4): #TODO this is no longer up to date, you should remove it in order to put new conversation flow AI
    template = """You are a classifier AI. each object you are given is a memory of a specific subject cluster and they need to be grouped. your job is to identify each object inside the following cluster and make a summary of the cluster as a whole. your answer must follow this format:
    Cluster name = given cluster name, for example: how to build an oxygen sensor
    
    Memories {1, 3, 4, 17 and 19}: These memories talk about the electronics needed to build an oxygen sensor
    Memories {2,6 and 14}: These memories talk about the manufacturing challenges of building an oxygen sensor
    Memories {5,9,10,11 and 13}: These memories talk about the materials needed to build an oxygen sensor
    
    CLUSTER SUMMARY
    This Cluster explains how oxygen sensors are built in a manufacturing plant. it highlights the electronics needed, the most common challenges and the materials, as well as their cost, needed to build an oxygen sensor.
    END SUMMARY
    
    The user will now ask you to format a cluster:
    
    """

    formatted_text = []
    for index, item in enumerate(text, 1):
        formatted_string = f"{index}. {str(item)}"
        formatted_text.append(formatted_string)
    formatted_string = "\n".join(formatted_text)
    formatted_string = f"Cluster name:<{reflection_name}>"+ "\n" + formatted_string
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": f"{template}"},
            {"role": "user", "content": f"{formatted_string}"}
        ],
        temperature=0.2
    )["choices"][0]["message"]
    AI_trainer.training_dataset_creation(formatted_string, response, "Cluster_reflection")
    return response
@retry_on_openai_error()
def notetaking_AI(text, model="gpt-3.5-turbo-16k"):#needs to get a list of texts #TODO this is no longer up to date, you should remove it in order to put new conversation flow AI
    # Convert text to a list if it's not already
    if not isinstance(text, list):
        text = [text]
    templateNotetakingLogicSingleText = """You are a student trying to create a syllabus from your textbook. The following text is a chapter from your textbook. Use the Cornell method to identify the main points and lessons. The Cornell method focuses on:

General Notetaking: Capture actionable insights and lessons from the text. Lessons should be universal principles or advice that can be applied in various contexts, not just specific to the author's experience. Avoid including mere observations.
Main Ideas: Extract the most crucial points from the text. Delve deeper into the content, considering both the explicit and implicit messages the author is conveying.
Summary: Provide a concise paragraph summarizing the essence of the chapter.
Concepts: Highlight foundational ideas or principles introduced in the chapter. Concepts should be overarching themes or theories, not just specific facts or examples from the author's life.
For clarity:

A lesson might be: "The importance of perseverance in the face of failure."
An observation might be: "The author failed multiple times before succeeding."
A concept might be: "The iterative process of product development."
A fact might be: "The author launched the iPod in 2001."
Structure your response as follows:

Chapter [insert number]: [insert insightful chapter title]

START OF GENERAL NOTETAKING
Lessons:

[Bullet point 1]
[Bullet point 2]
...
END OF GENERAL NOTETAKING
START OF MAIN IDEAS
Main points:

[Bullet point 1]
[Bullet point 2]
...
END OF MAIN IDEAS
START OF SUMMARY
Summary:
[Concise paragraph]
END OF SUMMARY

START OF CONCEPT
Concepts:

[Bullet point 1]
[Bullet point 2]
...
END OF CONCEPT
Now, based on the following text from the textbook, structure your notes with depth and precision:"""

    # templateNotetakingLogicMultipleTexts = "You are a student trying to create a sort of syllabus from your textbook. The following text is a part of a chapter from your textbook. Here are your previous notes on this chapter that you can build upon [previous notes variable]. Identify the main points and lessons using the Cornell method and give a name to the chapter you are reading in order to facilitate further reading. the Cornell method is when you focus on three parts of notetaking, the notetaking, usually they are definitions and contain the bulk of the useful information, the cues, which are keywords and central ideas with a bullet point format and finally the summary, which is a short paragraph that gives a summary of everything. Text:"
    label = ""
    template = ""
    content= ""
    previous_notes=[]
    for i, item in enumerate(text):
        if i == 0:
            template = templateNotetakingLogicSingleText
            label = "Notetaking_AI_single_text"
            content = str(text)
        # else:
        # template = templateNotetakingLogicMultipleTexts.replace("[previous notes variable]", previous_notes)
        # label = "Notetaking_AI_multiple_text"
        # content=item

        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": template},
                {"role": "user", "content": content}

            ],
            temperature=0.2,
            max_tokens=700
        )

        answer = response['choices'][0]['message']['content'].strip()

        # If the second template is used, the item variable should be an addition of the previous_notes and then of item
        # if i != 0:
        # item = previous_notes + "\n" + item

        AI_trainer.training_dataset_creation(item, answer, label)

        previous_notes.append(answer)  # append the answer to the previous notes

    return previous_notes

@retry_on_openai_error()
def spiky_AI(context, previous_exchanges="", model="gpt-4_8k"): #TODO this is no longer up to date, you should remove it in order to put new conversation flow AI
    """
    Generate a response using the Spiky AI based on the given context and previous exchanges.

    Parameters:
    - context (str): The main input or conversation context.
    - previous_exchanges (str): Previous conversation exchanges.
    - model (str): The OpenAI model to be used.

    Returns:
    - str: The generated response.
    """

    # Spiky's personality and behavior template (under 75 words)
    spiky_template = """Imagine an AI with a coaching and consultative personality, skilled in the Socratic method, SMART goal-setting and SWOT analysis. This AI is designed to guide users through in-depth exploration of their ideas, addressing uncertainties, and strategizing for future ventures or simple projects of any kind. Don't hesitate to give insightful comments and suggestions to help the user discover new possibilities.Engage in a conversation with a user about their aspirations."""

    # Construct the total context
    context_total = " Previous exchanges: " + previous_exchanges + ". User input: " + context

    # Generate a response using OpenAI API
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": spiky_template},
            {"role": "user", "content": context_total}
        ],
        max_tokens=500
    ).choices[0].message['content'].strip()

    # Bonus question creation is disabled
    # bonus_questions_and_context = create_next_question_or_answer(context + previous_exchanges + response)
    # bonus_questions = bonus_questions_and_context[0]
    # bonus_questions_str = '\n'.join(bonus_questions)

    # Create a training dataset
    AI_trainer.training_dataset_creation(context_total, response, "spiky_AI_conversation")

    # Return the final response
    return response  # + ". Here are some bonus questions for the user to answer: " + bonus_questions_str

@retry_on_openai_error()
def similarity_approval_AI(question, text, model="text-davinci-003"):
    # Convert list to string if needed
    if isinstance(question, list):
        question = ' '.join(question)
    if isinstance(text, list):
        text = ' '.join(text)

    # New template prompt asking the model to evaluate if the question or concept is related to the text
    template = f"Do you agree that the question or concept < {question} > is strongly related to the following text? Please answer with 'true' or 'false'. "

    # Get token IDs for "true" and "false"
    true_token_id = get_token_id("true", model)
    false_token_id = get_token_id("false", model)

    # Make the API call with logit_bias and max_tokens set
    response = openai.Completion.create(
        model=model,
        prompt=template + text,
        max_tokens=1,
        logit_bias={str(true_token_id): 100, str(false_token_id): 100}
    ).choices[0].text.strip()
    response=response
    AI_trainer.training_dataset_creation(question+text,response,"similarity_approval_AI")
    approval = logic_gate(response)

    return approval

@retry_on_openai_error()
def goal_approval_AI(text, model=TEXT_DAVINCI_COMMON_3):
    # New template prompt asking the model to evaluate if the input text is a valuable goal
    template = """Based on the following information, determine if this represents a valuable goal that is worth pursuing. The goal is:

    {}

    Is this a valuable goal worth pursuing? Please answer with 'true' or 'false'. """.format(text)

    # Get token IDs for "true" and "false"
    true_token_id = get_token_id("true", model)
    false_token_id = get_token_id("false", model)

    # Make the API call with logit_bias and max_tokens set
    response = openai.Completion.create(
        model=model,
        prompt=template,
        max_tokens=1,
        logit_bias={str(true_token_id): 100, str(false_token_id): 100}
    ).choices[0].text.strip()
    AI_trainer.training_dataset_creation(text, response, "goal_approval_AI")
    # Apply the logic gate function to determine approval
    approval = logic_gate(response)

    return approval


@retry_on_openai_error()
def goal_creation_AI(text,model=TEXT_DAVINCI_COMMON_3):
    template = """Based on the brief overview of the client conversation (between Spiky, our agent and the User, our client), please distill the key information into a single, concise bullet point for the relevant section.
     This single bullet point(or -) should summarize the most important detail or action item discussed, making it easy to understand the essence of that topic. here are multiple examples to better understand:
     -The user thinks that a supercooler is necessary to produce meaningful revenue
     -Each individual factory should not cost more than 20 000 dollars
     -He recruited a total of 5 workers to build the prototype
     
     client conversation:"""
    response = openai.Completion.create(
        model=model,
        prompt=template + text + "what is your take on the previous conversation, give your answer in a single bullet point",
        max_tokens=50,
        temperature = 0.35
    ).choices[0].text.strip()
    AI_trainer.training_dataset_creation(text, response, "goal_creation_ai")
    return response
import re

@retry_on_openai_error()
def create_new_section_AI(text, model=TEXT_DAVINCI_COMMON_3):
    template = """Based on the single exchange you will receive between our agent and the client (Spiky and the User respectively), your task is to identify the topic being discussed and create a new section name that encapsulates that topic.
     The section name should be concise yet descriptive enough to give a clear idea of what the section will cover and be general enough to allow further, similar exchanges to fit in the same section.
     This is an example that showcases multiple section names for further reference, remember that there is no need to label them as sections, you just need to name them with this type of short format: Budgeting, Project Timeline, Team and Labor, Objectives, Constraints and Challenges, Technical Specifications For Hydroponic Farm, Energy and Sustainability, Technical specifications for crops"""

    response = openai.Completion.create(
        model=model,
        prompt=template + text,
        max_tokens=50,
        temperature=0.1
    ).choices[0].text.strip()
    response = response.lower()
    # Remove specific substrings
    for word in ["core topic:", "section name:", "topic:"]:
        response = response.replace(word, "")

    # Remove any extra spaces
    response = re.sub(' +', ' ', response).strip()

    AI_trainer.training_dataset_creation(text, response, "create_new_section_ai")

    return response

def function_calling(messages, function_manager, model=FUNCTION_CALLING_GPT4):
    if not isinstance(messages, list):
        messages = []

    function_items = function_manager.functions_dict.items()

    function_json_schema = [{'name': name, **info['metadata']} for name, info in function_items]

    function_dict = {name: info['function'] for name, info in function_items}

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        functions=function_json_schema,
        function_call="auto"
    )

    response_message = response["choices"][0]["message"]

    function_json = response_message["function_call"]
    function_name = function_json["name"]
    function_args = json.loads(function_json["arguments"])

    function_to_call = function_dict[function_name]
    function_response = function_to_call(**function_args)
    function_response_string = convert_to_string(function_response)

    messages.append(response_message)
    messages.append(
        {
            "role": "function",
            "name": function_name,
            "content": function_response_string,
        }
    )

    second_response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )
    print(second_response)
    return function_to_call, function_json, function_name, function_args

@retry_on_openai_error()
def conversation_comment_AI(text,model=TEXT_DAVINCI_COMMON_3):#this function could potentially write better comments by having interactions with the wole document or interact with other AIs
    template="""You will receive a text and your goal is to see if there are incoherences and point them out or simply what are your thaughts on the ides presented. If you see something that is not feasable or seems to go in the wrong directions you can also point it out and explain why."""
    template_retrospection="""You will receive a text along with any previous comments or analyses that have been made. Your goal is to scrutinize the text for incoherences, feasibility, and direction. Use the following format to structure your response:
- **Copy of orignal comments: Rewrite the orignal comments, you can slightly modify it if it looks helpful by adding details.
- **Initial Summary**: Provide a brief summary of your overall impressions of the text. 50 words
  
- **Bullet Points**: 
  - **Inconsistencies**: Point out any inconsistencies or non-probable statements that you find and explain why they are problematic or misleading.
  - **Feasibility**: Identify elements that may not be feasible and explain why.
  - **Direction**: Comment on the overall direction of the text. Is it aligned with its stated goals or does it veer off course?
  
- **Integration with Previous Comments**: Consider the previous comments and analyses. Integrate your current insights with the existing comments to provide a comprehensive understanding.
  - **Agreements**: Note where your analysis aligns with previous comments.
  - **Disagreements**: Point out where you differ and explain why.
  - **Solutions**:Point out what are the best solutions to explore. If there are no problems, try to see how to optimize what is already good
  
- **Final Thoughts**: Conclude with any additional observations or recommendations for improvement. 100 words, small essay format

Please ensure that your analysis is comprehensive, integrating your current insights with previous comments for a complete overview. It is important to point out the missing information you could use to make a more complete analysis. Even if you think obtaining the information is complex, as long as it can be beneficial
"""
    response = openai.Completion.create(
        model=model,
        prompt=template + text,
        max_tokens=150,
        temperature=0.8
    ).choices[0].text.strip()
    retrospection_text = response + "\noriginal comments above" + text

    response = openai.Completion.create(
        model=model,
        prompt=template_retrospection + retrospection_text,
        max_tokens=400,
        temperature=0.8
    ).choices[0].text.strip()

    AI_trainer.training_dataset_creation(text, response, "conversation_comments_ai_2")
    return response + "\n\n\n"

def create_code(task: dict, outside_dependencies: dict, advice: str = None, lessons: str = None, model=CHAT_GPT4) -> str:
    system_prompt_programmer = f"""Your task is to create a Python function based on the following requirements:
    - Description: {task['description']}
    - Input Type: {task['input_type']}
    - Output Type: {task['output_type']}
    - Allowed Libraries: {', '.join(outside_dependencies['allowed_libraries'])}
    - Disallowed Functions: {', '.join(outside_dependencies['disallowed_functions'])}
    - Advice: {advice if advice else 'None'}
    - Lessons: {lessons if lessons else 'None'}

    Please provide the following:
    - Code Snippet: 
    - Used Libraries: 
    - Complexity Metric: 
    - Coding Standards: 
    - Resource Metrics: """

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
                {"role": "system", "content": "You are a programmer tasked with writing Python code. Always add a small analysis paragraph before the code to explain how you will approach the problem"},
                {"role": "user", "content": system_prompt_programmer}
            ],
        temperature=0.7
        )["choices"][0]["message"]
    trainer_prompt=f"""
Description: {task['description']}
Input Type: {task['input_type']}
Output Type: {task['output_type']}
Allowed Libraries: {', '.join(outside_dependencies['allowed_libraries'])}
Disallowed Functions: {', '.join(outside_dependencies['disallowed_functions'])}
Task Specific Advice:{advice if advice else 'None'}"""
    #AI_trainer.training_dataset_creation(trainer_prompt,response,"Programmer_AI")
    return response

def evaluate_and_advise(programmer_output: dict, task: dict, outside_dependencies: dict, model=CHAT_GPT4) -> str:
    # Prepare the enriched system prompt for GPT-4
    system_prompt = f"""Please evaluate the following code snippet based on these criteria and constraints:

    ### Criteria:
    1. **Code Correctness**: Does the code perform the task as described in the task description?
    2. **Library Use**: Are the libraries used appropriate, and within the allowed list?
    3. **Code Complexity**: Is the code's complexity metric reasonable for the task?
    4. **Coding Standards**: Does the code adhere to common coding standards like PEP 8?
    5. **Resource Efficiency**: Are the resource metrics within acceptable limits?

    ### Constraints:
    - **Task Description**: {task['description']}
    - **Input Type**: {task['input_type']}
    - **Output Type**: {task['output_type']}
    - **Library Constraints**: Allowed Libraries - {', '.join(outside_dependencies['allowed_libraries'])}, Disallowed Functions - {', '.join(outside_dependencies['disallowed_functions'])}
    - **Data Constraints**: {outside_dependencies['expected_data']}

    ### Code Snippet Details:
    - **Code Snippet**: {programmer_output['code']}
    - **Used Libraries**: {', '.join(programmer_output['used_libraries'])}
    - **Complexity Metric**: {programmer_output['complexity_metric']}
    - **Coding Standards**: {programmer_output['coding_standards']}
    - **Resource Metrics**: Memory - {programmer_output['resource_metrics']['memory']}, CPU - {programmer_output['resource_metrics']['CPU']}

    Is the code satisfactory based on these criteria and constraints? If not, please provide specific advice and lessons for improvement."""

    # GPT-4 API call
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
                {"role": "system", "content": "You are a code assessor tasked with evaluating Python code. When problems are encoutnered they are highlighted and simple ameliorations are also mentioned in you answers."},
                {"role": "user", "content": system_prompt}
                ],
        temperature=0.2
        )["choices"][0]["message"]
    trainer_prompt = f"""
    ### Constraints:
    - **Task Description**: {task['description']}
    - **Input Type**: {task['input_type']}
    - **Output Type**: {task['output_type']}
    - **Library Constraints**: Allowed Libraries - {', '.join(outside_dependencies['allowed_libraries'])}, Disallowed Functions - {', '.join(outside_dependencies['disallowed_functions'])}
    - **Data Constraints**: {outside_dependencies['expected_data']}

    ### Code Snippet Details:
    - **Code Snippet**: {programmer_output['code']}
    - **Used Libraries**: {', '.join(programmer_output['used_libraries'])}
    - **Complexity Metric**: {programmer_output['complexity_metric']}
    - **Coding Standards**: {programmer_output['coding_standards']}
    - **Resource Metrics**: Memory - {programmer_output['resource_metrics']['memory']}, CPU - {programmer_output['resource_metrics']['CPU']}"""
    #AI_trainer.training_dataset_creation(trainer_prompt,response,"Code_Assessor_AI")
    return response

logging.basicConfig(filename=r'C:\Users\philippe\Documents\pdf to txt files\logs\application.log', level=logging.ERROR,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
@retry_on_openai_error()
def get_embedding(text, model=TEXT_EMBEDDING_ADA):
    placeholder_text = "placeholder"
    try:
        if isinstance(text, list):
            text = ' '.join(text)
        text = text.replace("\n", " ")
        return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
    except Exception as e:
        logging.error(f"get_embedding: an error occurred - {str(e)}")
        return openai.Embedding.create(input = [placeholder_text], model=model)['data'][0]['embedding']

def convert_to_string(input_value):
    try:
        # Attempt to convert the input to a string
        return str(input_value)
    except Exception as e:
        # Handle the exception and return an error message
        return f"Failed to convert to string due to: {e}"
