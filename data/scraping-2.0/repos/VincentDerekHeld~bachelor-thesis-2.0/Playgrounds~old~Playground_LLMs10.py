import os

import requests
from openai import OpenAI
import spacy

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def preprocess_text_with_LLM(doc):
    debug_mode = True

    text_contains_listings = False
    for token in doc:
        if token.tag_ == "LS":
            text_contains_listings = True
            break

    relevance_of_sentence = True

    transform_implict_actions = True

    intro = """
    #### Intro: ###
    You carefully follow the instructions. You solve the tasks step by step keeping as many words from the input text as possible, and never add information from the examples to the output.\n"""

    outro = """\n ### TEXT ### \n"""
    answer_outro = "\n ### Answer / Response: ###"

    prompts = []
    inital_prompt = ("""
    ### Instruction: ###
    Read the text carefully and try to understand the content. 
    Return on this message an empty message (just a spaces without any other characters).
    ### Full Text ### \n""")

    second_prompt = ("""
    ### Example: ###
    # Example Input: # 
    The actor shall determine:
    a) what needs to be done..
    b) the methods
    c) feedback on:
        1) the results
        2) monitoring
    
    # Example Output: # 
    The actor shall determine what needs to be done...
    The actor shall determine the methods...
    The actor shall determine feedback on the results, monitoring, 
    
    ### Instruction: ####
    Keep as many words from the original text as possible, and never add information (from the example) to the output. 
    Return the listings carefully transformed into a continuous text based on the example and filter the bullet points.
    
    """)

    filter_outro = """
    ### Instruction: ###
    1) Decide carefully based on the provided background if a part of the following sentence fulfills the conditions of the provided background information. If the condition is fulfilled, go to 2), else go to 3).
    2) Filter carefully the information, that fulfills the condition, from the text (but sill return full sentences) or if the sentence consists only out of this irrelevant information return an empty message (just a spaces without any other characters).
    3) Return the text carefully without any changes or interpretations.
    """

    # Relevance of Sentence
    if relevance_of_sentence: prompts.append(""" 
    ### Background Information: ###
    Introduction sentence that describe what the company does in general are not relevant and must be filtered.
        Example: The Sentence "A small company manufactures customized bicycles." must be filtered.
    """ + filter_outro)

    if relevance_of_sentence: prompts.append(""" 
    ### Background Information: ###
    Information that describes the outcome or the goal of the process or an activity are not relevant and must be filtered.
                Example: "to ensure valid results." this part must be filtered.
                Example: "to ensure that the audit programme(s) are implemented and maintained." this part must be filtered.
                Example: "as evidence of the results." this part must be filtered. 
                Example: "that are relevant to the information security management system" this part must be filtered.
    """ + filter_outro)
    #Information that describes the reason of an activity must be filtered.

    if relevance_of_sentence: prompts.append(""" 
    ### Background Information: ###
    Information that describes the decision criteria for methods or techniques are not relevant and must be filtered.
                Example: "The methods selected should produce comparable and reproducible results to be considered valid;" must be filtered.
                    -> The sentence consists only out of this irrelevant information and an empty message must be returned.
    """ + filter_outro)

    if relevance_of_sentence: prompts.append(""" 
    ### Background Information: ###
    Information that clarifies that something is not universally applicable are not relevant and must be filtered.
                Examples: "as applicable", "if applicable", "where applicable" must be filtered.
       """ + filter_outro)

    if relevance_of_sentence: prompts.append(""" 
    ### Background Information: ###
    Sentence parts that contain examples are not relevant and must be carefully filtered from the sentence.
                Example of key words:  parts containing "for example ...", "e.g. ..." must be filtered.
    """ + filter_outro)

    if relevance_of_sentence: prompts.append(""" 
    ### Background Information: ###
        References to other Articles or Paragraphs are not relevant and must be filtered.
                Example: "referred to in Article 22(1) and (4)" must be filtered
                Example: "in accordance with Article 55"
    """ + filter_outro)

    # Implicit Actions
    if transform_implict_actions: prompts.append("""
    Some sentences contain implicit actions. Implicit or implied actions are actions that are actions that are not explicitly mentioned in the sentence, but can be inferred from the sentence. 
    Implicit actions are often formed by using the participle.
            #Example 1: „Documented information shall ...“ -> „They shall document the information...“
            #Example 2: "They then submit an order ticket to the kitchen to begin preparing the food." -> „They submit an order ticket to the kitchen. The kitchen prepares the food.“
        
        ### Instruction: ####
        Solve the tasks carefully step by step. Do not comment on any of the steps. Return only the (transformed) sentence without interpretations.
        1) Analyze the following sentence to determine if it contains any implicit actions. If it contains implicit actions go to 2), else go to 3). Do not comment on this step.
        2) If implicit actions are identified, these must be carefully converted into explicit actions following the following conditions, else return the sentence without any changes. 
            1. Condition: The original structure and order of the sentence must be retained as far as possible.
            2. Condition: The original wording of the sentence must be retained as far as possible.
        3) Do not comment on this step and return the input sentence without any changes and interpretations. 
    """)

    if False:
        # Active Voice
        prompts.append("""
        ### Background Information: ###
        Example 1:
            Active Voice: "Whenever they receive an order, ..."
            Passive Voice: "... a new process instance is created."
            -> New Sentence: "Whenever they receive an order, they create a new process instance."
        Example 2:
            "If it is not available, it is back-ordered."
            Active Voice: "If the product is not available"
            Passive Voice: "it is back-ordered."
            -> To determine the active voice, the actor must be identified in the text from the inital request. New Sentence: "If the product is not available, they department back-orders the product."
        
        Somtimes for the transformation the actor is needed, which is not given in the sentence. Therefrom the actor must be identified in the text from the inital request.
        ### Instruction: ####
        If a part of a sentence is in passive voice, return this part of the sentence transformed into an active voice sentence (inlcuding the other parts), else return the sentence without any changes.
        If the actor is not given in the sentence, identify the actor in the text from the inital request and do not use information or verbatim from the examples.
        """)

    # TODO
    # Reference Resolution
    if False: prompts.append("""
    Using the text provided, replace all pronouns with the specific names of the people or objects they refer to. 
    Do not add, infer, or assume any information beyond what is explicitly mentioned in the text. 
    If a pronoun's antecedent is not clear from the text, leave the pronoun as is and return the sentence without any changes and interpretations.
    Please focus solely on direct replacements for clarity and accuracy.
    """)
    if False: prompts.append("""
    ### Instruction: ####
    1) Remember the whole text from the first request and resolve references in the sentence such as "she", "he", "it", "they" with the name of the ACTOR(s) from the text, if given.
    2) Remember the whole text from the first request and resolve references in the sentence such as "this", "that", "these", "those" with the name of the OBJECT(s) from the text, if given.
    3) Remember the whole text from the first request and resolve references in the sentence such as "another" with the name of the ACTOR(s) or OBJECT(s) from the text, if given.
    If no reference is given or the reference cannot be resolved, return the sentence without any changes and interpretations.
    """)
    if False: prompts.append("""
    Please restructure the provided sentences to fit the specified format: '[ACTOR] [MODAL VERB if present] [VERB in active voice] [OBJECT].' 
    Remember that an ACTOR can be a person, a group (like a company or department), or a representative entity (like 'Kitchen' for kitchen staff). 
    Do not confuse objects with actors; for instance, 'auditors' in 'Select the auditors' are objects, not actors. K
    eep modal verbs such as 'must', 'shall', 'should', 'can' in their original form within the sentence. 
    Do not introduce new information; use only the actors and objects as they are presented in the text. 
    Your focus is on reformatting sentences without altering their original meaning or adding content.
    """)
    if False: prompts.append("""
    ### Background Information ###
    1. Actors are the subject of a sentence, or the person or thing that performs the action of the verb
        For the identification of the ACTOR, keep in mind, a actor can  for e.g. be a natural person, a organization, such as a company or a department, but sometimes also a place, a device or a system can be an valide Actor.
            Example: The "Kitchen" represents the "kitchen staff", and therefore the "Kitchen" can be an actor.
        Make sure not to use Objects as Actors:
            Example:  In the sentence "Select the auditors" the "auditors" are the object, not the actor.
            
    2. Modal verbs have to stay with the original format  in the sentence.
            -> Modal verbs, are verbs that express the strength of an expression. 
            Example modal verbs are: "must", "shall", "should", "can".
                Example 1: In the sentence "The organization shall determine..." the modal verb "shall" has to stay in the sentence.
    ### Instruction: ###
    Restructure every sentence to achieve the following structure of the sentence: "[ACTOR] [MODAL VERB if it exists in the sentence] [VERB in active] [OBJECT]."
    """)
    if False: prompts.append("""
    Transform the given sentences by following specific structural rules. First, identify placeholders such as '[ACTOR]', '[CONDITION]', '[OBJECT]', and '[ACTION]'. Then, replace these placeholders with appropriate words extracted from the context. Once you have the necessary components, restructure the sentences as per the following guidelines:
    If there is an action that does not depend on a condition, structure the sentence using the active voice in this format: '[ACTOR] [MODAL VERB if present] [VERB in active voice] [OBJECT]'.
    If an action is conditional, reformat the sentence to: 'If [CONDITION], [ACTOR] [ACTION], ELSE [alternate ACTION if provided in the text]'.
    When two actions are contingent on the same condition, combine them into a single sentence: 'If [CONDITION], [ACTOR] [ACTION], ELSE [alternate ACTION if mentioned]'.
    For example, take the sentences 'If the controller becomes aware of a personal data breach, the controller shall notify the supervisory authority within 72 hours.' and 'If the notification isn't made within 72 hours, the controller shall provide reasons for the delay.' Merge them into 'If the controller becomes aware of a personal data breach, the controller shall notify the supervisory authority within 72 hours, else the controller shall provide reasons for the delay.', as they follow an IF-ELSE structure.
    Please apply these rules to restructure sentences accurately and clearly, maintaining the original meaning and intent of the text.
    """)
    if False:
        if use_all_prompts: prompts.append("""
                         Carefully replace all placeholders, such as "[ACTOR], [CONDITION], [OBJECT], [ACTION]" with the right words from the text and restrcuture the sentence in the follwing way:
                         
                         1) If the action is not based on a condition restructure the sentence with the following structure: "[ACTOR] [MODAL VERB if exists in the sentence] [VERB in active] [OBJECT]."
                         
                         2) If the action is based on a condition restructure the sentence with the following structure: „If [CONDITION], [ACTOR] [ACTION], ELSE [other ACTION, if in the Text]“. 
                        
                         3) If two ACTIONs are based on the same CONDITION merge the sentences to the following structure: „If [CONDITION], [ACTOR] [ACTION], ELSE [other ACTION, if in the Text]“. 
                                Example: 
                                The Sentence „If the controller becomes aware of a personal data breach, the controller shall notify the supervisory authority within 72 hours.“ and
                                the Sentence „If the notification isn't made within 72 hours, the controller shall provide reasons for the delay.“ can be merged into one Sentence „If the controller becomes aware of a personal data breach, the controller shall notify the supervisory authority within 72 hours, else the controller shall provide reasons for the delay.“, as they are based on a IF - ELSE structure.
                """)

        if False: prompts.append("""
                Ensure carefully that all placeholders, such as "[ACTOR], [CONDITION], [OBJECT], [ACTION]" have been replaced with the correct actors, condtions, objects and actions from the text and the text does not contain any [].
                   """)

        if False: prompts.append("""
                        
                        If the CONDITION part is at the end of the sentence, move it to the front of the sentence. 
                        Example: The Sentence "The data subject has the right to have their personal data transmitted directly from one controller to another, if technically feasible." can be restructured into "If technically feasible, the data subject has the right to have their personal data transmitted directly from one controller to another.".
                        ###### TEXT ###### \n
                """)

    def generate_response(prompt) -> str:
        try:
            if debug_mode: print(f"*** Prompt: *** len: {len(prompt).__str__()} \n {prompt} \n")
            #model = "gpt-3.5-turbo"
            model = "gpt-3.5-turbo-instruct"
            response = client.chat.completions.create(
                model= model,
                messages=[
                    {"role": "system", "content": intro},
                    {"role": "assistant", "content": ""},
                    {"role": "user", "content": prompt},
                ]
            )
            response_text = response.choices[0].message.content
            response_text = response_text.strip()
            if debug_mode:
                print(f"*** Response: ***\n {response_text}")
                print("*" * 50)
            return response_text

        except requests.exceptions.Timeout:
            # Handle timeout specifically
            if debug_mode:
                print("Request timed out")
            return "Request timed out"

        except Exception as e:
            # Handle any other exceptions
            if debug_mode:
                print(f"An unexpected error occurred: {e}")
            return f"An unexpected error occurred: {e}"

    # Initital Instructions and
    result = ""
    generate_response(inital_prompt + doc.text)
    print(f"Text contains listings: {text_contains_listings}")
    if text_contains_listings:  #
        new_text = generate_response(second_prompt + outro + doc.text + answer_outro)
        nlp = spacy.load('en_core_web_trf')
        doc = nlp(new_text)
        output_path = f"/Evaluation/GPT-Text/temp.txt"
        with open(output_path, 'w') as file:
            file.write(new_text)
        print(f"Created Text Temp File:")

    for number, sent in enumerate(doc.sents):
        # print(f"**** Sent. {number}: {sent.text}")
        current_sent: str = sent.text
        if current_sent.isspace() or current_sent.__len__() == 0:
            print(f"Skipped on Sent No. {number} because only whitespace")
            next(doc.sents)
        for prompt in prompts:
            print(f"current_sent: {current_sent}")
            print(f"current_sent.isspace(): {current_sent.isspace().__str__()}")
            print(f"current_sent.__len__(): {current_sent.__len__()}")
            if current_sent.isspace() or current_sent.__len__() == 0:
                print(f"Sent No. {number} has been returned as empty message.")
                next(doc.sents)
                break
            # intro
            query = prompt + outro + current_sent + answer_outro
            current_sent = generate_response(query)

        result = result + "" + current_sent + "\n"

    print("**** Full description: **** \n" + result.strip().replace("\n", " ").replace(".", ". ").replace("!",
                                                                                                          "! ").replace(
        "?", "? "))
    return result.strip()


def write_to_file(number: int, nlp):
    input_path = f"/Users/vincentderekheld/PycharmProjects/bachelor-thesis/project/Text/text_input_vh/Text{number.__str__()}.txt"
    text_input = open(input_path, 'r').read().replace("\n", " ")
    doc = nlp(text_input)
    for sent in doc.sents:
        print(f"Sent: {sent.text}")
    content = preprocess_text_with_LLM(doc)
    output_path = f"/Users/vincentderekheld/PycharmProjects/bachelor-thesis/Evaluation/GPT-Text/Text{number.__str__()}-11.txt"
    with open(output_path, 'w') as file:
        file.write(content)
    print(f"Created Text: {number.__str__()}")


nlp = spacy.load('en_core_web_trf')
write_to_file(6, nlp)
