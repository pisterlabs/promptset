import os
import openai
import spacy

openai.api_key = os.environ["OPENAI_API_KEY"]

def preprocess_text_with_LLM(doc):
    # generate a response of a llm (open-ai) given a prompt
    debug_mode = True
    use_all_prompts = True
    Filtering = True

    intro = """#### Intro: ### \n 
    You are a system analyst who strictly and carefully follows the instructions. 
    You return carefully only the current sentence as a complete sentence without adding any extra information, interpretation, explanation, numerations, listings, or "->".
    Make sure that the previously given instructions are still followed.
    Keep as many words from the original text as possible, and never add information from the examples to the output.
    You handle every task sentence for sentence.\n"""

    instruction_headline = """"""

    outro = """Return only the transformed input without any additional information or annotations.
            ###### TEXT ###### \n"""

    answer_outro = "\n #### Answer / Response: #### \n"

    prompts = []
    inital_prompt = (
        """### Instruction: #### \n Read the text carefully and try to understand the content. 
        On this message, you return an empty message.
        ### Full Text ### \n""")

    # Relevance of Sentence

    if use_all_prompts: prompts.append(
        """
        ### Background Information ###
        Introductions that describe the goal of the process or what the company produces are not relevant.
            Example: "to ensure valid results. "
                        -> "The organization shall determine the methods for monitoring, measurement, analysis and evaluation, to ensure valid results" -> "The organization shall determine the methods for monitoring, measurement, analysis and evaluation"
            Example: "as evidence of the implementation of the audit programme(s)"
                        -> "Documented information shall be available as evidence of the implementation of the audit programme(s) and the audit results." -> "... shall document the results."
        
        ### Instruction: #### \n
        If the following sentence contains the described information, filter it from the sentence and return the sentence without this information, else return the input sentence.
        """)

    if False:
        if use_all_prompts: prompts.append("""
            Filter information from the text (initial query), that is not relevant for the process. The only information we are interested in are real process steps.
            Information that just address that the process starts or is finished are not relevant.
                Example: "The process instance is then finished."
            """)

        if use_all_prompts: prompts.append("""
                Filter information from the text (initial query), that is not relevant for the process. The only information we are interested in are real process steps.
                    Information that clarifies that something is not universally applicable are not relevant.
                        Example: "as applicable"
                                "The organization shall determine the methods for monitoring, measurement, analysis and evaluation, as applicable," -> The organization shall determine the methods for monitoring, measurement, analysis and evaluation"
                """)

        if use_all_prompts: prompts.append("""
                Filter information from the text (initial query), that is not relevant for the process. The only information we are interested in are real process steps.
                Examples are not relevant for the process.
                """)

        if use_all_prompts: prompts.append("""
                Filter information from the text (initial query), that is not relevant for the process. The only information we are interested in are real process steps.
                References to other Articles or Paragraphs are not relevant for the process.
                        Example: "referred to in Article 22(1) and (4)" can be filtered
                        Example: "in accordance with Article 55"
                """)

        if use_all_prompts: prompts.append("""
                Filter information from the text (initial query), that is not relevant for the process. The only information we are interested in are real process steps. 
                Including parts of sentences are not relevant for the process.
                Example: "The organization shall determine what needs to be monitored and measured, including information security processes and controls" -> "The organization shall determine what needs to be monitored and measured"
                """)

    if False:
        prompts.append("""
                Transform listings based on the structure of the text into a continuous text.
                ###### TEXT ###### \n""")

        # Active Voice
        if use_all_prompts: prompts.append("""
                Return every sentence transformed into an active voice sentence, without any additional information or annotations.
                ###### TEXT ###### \n """)

        # Implicit Actions
        if use_all_prompts: prompts.append("""
                Extract implicit ACTIONS from the Text at the end of the prompt and transform the Actions into explicit Actions.
                ## Examples: ##
                    #Example 1: The Sentence „Documented information shall be available as evidence of the implementation of the audit programme(s) and the audit results.“ must be transformed into „Documented information shall be available as evidence of the implementation of the audit programme(s) and the audit results.“ because it implies the ACTION „document the results“.
                    #Example 2: The Sentence "The Room Service Manager then submits an order ticket to the kitchen to begin preparing the food." must be transformed into „The Room Service Manager submits an order ticket to the kitchen. The kitchen prepares the food.“ and here it is importent to keep the order of the sentences and to explicitly name the ACTOR and the ACTION.
                ### 5. Instruction: #### \n
                Without copying verbatim from the provided examples or annotations such as "Result" or "Answer" or "Response", please return every sentence from the following Text transformed into an explicit Action.
                ###### TEXT ###### \n """)

        # Reference Resolution
        if use_all_prompts: prompts.append("""
            Remember the whole text from the first request and resolve references:
            Examples:
            -> Replace references such as "she, he, it, they" in the sentence with the name of the ACTOR(s).
            -> Replace references such as "this, that, these, those" in the sentence with the name of the OBJECT(s).
            -> Replace references such as "another" with the corresponding ACTOR(s) or OBJECT(s).  
            ###### TEXT ###### \n """)

        if use_all_prompts: prompts.append("""
                ### Background Information ###
                1. Actors are the subject of a sentence, or the person or thing that performs the action of the verb
                    For the identification of the ACTOR, keep in mind, a actor can  for e.g. be a natural person, a organization, such as a company or a department, but sometimes also a place, a device or a system can be an valide Actor.
                            Example: The "Kitchen" represents the "kitchen staff", and therefore the "Kitchen" can be an actor.
                    Make sure not to use Objects as Actors:
                            Example:  In the sentence "Select the auditors" the "auditors" are the object, not the actor.
    
                2. Modal verbs have to stay with the original format  in the sentence.
                -> Modal verbs, are verbs that express the strength of an expression. 
                    Example modal verbs are: "must", "shall", "should", "can".
                     "Example 1: In the sentence "The organization shall determine..." the modal verb "shall" has to stay in the sentence.
                ### Background Information End ###
                
                Restructure every sentence to achieve the following structure of the sentence: "[ACTOR] [MODAL VERB if it exists in the sentence] [VERB in active] [OBJECT]."
                ###### TEXT ###### \n """)

        if use_all_prompts: prompts.append("""
                 Carefully replace all placeholders, such as "[ACTOR], [CONDITION], [OBJECT], [ACTION]" with the right words from the text and restrcuture the sentence in the follwing way:
                 
                 1) If the action is not based on a condition restructure the sentence with the following structure: "[ACTOR] [MODAL VERB if exists in the sentence] [VERB in active] [OBJECT]."
                 
                 2) If the action is based on a condition restructure the sentence with the following structure: „If [CONDITION], [ACTOR] [ACTION], ELSE [other ACTION, if in the Text]“. 
                
                 3) If two ACTIONs are based on the same CONDITION merge the sentences to the following structure: „If [CONDITION], [ACTOR] [ACTION], ELSE [other ACTION, if in the Text]“. 
                        Example: 
                        The Sentence „If the controller becomes aware of a personal data breach, the controller shall notify the supervisory authority within 72 hours.“ and
                        the Sentence „If the notification isn't made within 72 hours, the controller shall provide reasons for the delay.“ can be merged into one Sentence „If the controller becomes aware of a personal data breach, the controller shall notify the supervisory authority within 72 hours, else the controller shall provide reasons for the delay.“, as they are based on a IF - ELSE structure.
                 ###### TEXT ###### \n
        """)

        if True: prompts.append("""
        Ensure carefully that all placeholders, such as "[ACTOR], [CONDITION], [OBJECT], [ACTION]" have been replaced with the correct actors, condtions, objects and actions from the text and the text does not contain any [].
        ###### TEXT ###### \n
           """)

    if False: prompts.append("""
            
            If the CONDITION part is at the end of the sentence, move it to the front of the sentence. 
            Example: The Sentence "The data subject has the right to have their personal data transmitted directly from one controller to another, if technically feasible." can be restructured into "If technically feasible, the data subject has the right to have their personal data transmitted directly from one controller to another.".
            ###### TEXT ###### \n
    """)

    def generate_response(prompt) -> str:
        # Get the response from GPT-3
        model_engine = "text-davinci-003"
        if debug_mode: print(f"*** Prompt:  ****  len: {len(prompt).__str__()} \n {prompt} \n \n")
        response = openai.Completion.create(
            engine=model_engine, prompt=prompt, max_tokens=1024, n=1, stop=None, temperature=0.0)
        # Extract the response from the response object
        response_text = response["choices"][0]["text"]
        text_response = response_text.strip()
        if debug_mode: print(f"*** Response:  **** \n {text_response} \n \n ")
        return text_response

    # Initital Instructions and
    result = ""
    generate_response(inital_prompt + doc.text)
    for number, sent in enumerate(doc.sents):
        # print(f"**** Sent. {number}: {sent.text}")
        current_sent: str = sent.text
        if current_sent.isspace():
            print(f"Skipped on Sent No. {number} because only whitespace")
            next(doc.sents)
        for prompt in prompts:
            query = intro + instruction_headline + prompt + outro + current_sent + answer_outro
            current_sent = generate_response(query)
        result = result + "" + current_sent + "\n"

    print("**** Full description: **** \n" + result.strip().replace("\n", " ").replace(".", ". ").replace("!",
                                                                                                          "! ").replace(
        "?", "? "))
    return result


def write_to_file(number: int, nlp):
    input_path = f"/Users/vincentderekheld/PycharmProjects/bachelor-thesis/project/Text/text_input_vh/Text{number.__str__()}.txt"
    text_input = open(input_path, 'r').read().replace("\n", " ")
    doc = nlp(text_input)
    content = preprocess_text_with_LLM(doc)
    output_path = f"/Users/vincentderekheld/PycharmProjects/bachelor-thesis/Evaluation/GPT-Text/Text{number.__str__()}-2.txt"
    with open(output_path, 'w') as file:
        file.write(content)
    print(f"Created Text: {number.__str__()}")


# input_path = "/Users/vincentderekheld/PycharmProjects/bachelor-thesis/project/Text/text_input_vh/Text7.txt"
# text_input = open(input_path, 'r').read().replace("\n", " ")
nlp = spacy.load('en_core_web_trf')
# doc = nlp(text_input)
# preprocess_text_with_LLM(doc)
write_to_file(8, nlp)
