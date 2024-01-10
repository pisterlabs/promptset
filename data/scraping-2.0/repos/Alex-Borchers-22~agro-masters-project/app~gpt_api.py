import openai
from app.gpt_config import DevelopmentConfig
import json

openai.api_key = DevelopmentConfig.OPENAI_KEY

def generateChatResponse(prompt):
    """
    Generates response from ChatGPT API call

    Parameters
    ----------
    prompt : String
        The prompt given to ChatGPT

    Returns
    -------
    answer : String
        Chat GPT response
    """

    messages = []
    messages.append({"role": "system", "content": "You are a helpful assistant."})

    question = {}
    question['role'] = 'user'
    question['content'] = prompt
    messages.append(question)

    response = openai.ChatCompletion.create(model='gpt-3.5-turbo',messages=messages)

    try:
        answer = response['choices'][0]['message']['content']
    except:
        answer = 'Error'
    
    print("GPT Answer: " + answer)

    return answer

def mutateHyperParameters(ml_model, mutated_list, class_list):
    """
    Get mutated hyperparameters

    Parameters
    ----------
    ml_model : ML_Class object
        ML class object (must have .ml_model attribute)
    mutated_list : List
        List of changes in hyperparameters throughout each generation of mutants
    class_list : List
        List of SVC() classifiers related to specific type of model

    Returns
    -------
    ml_model : ML_Class object
        Updated ML Model with new hyperparameters set
    """

    # Create prompt to get 1 parameter
    #prompt = "Randomly pick one string from this list. All previous prompts should have no influence on your decision. Only reply with the string chosen: C, cache_size, class_weight, coef0, degree, gamma, kernel, max_iter, shrinking, tol, verbose"
    
    # Create Chat GPT prompt that just returns the new model classifier
    # Prompts attempted
    #prompt = 'This is a SVC() model trained via sklearn from the following library "from sklearn import svm". Randomly change or add one hyperparameter, do not remove any existing hyperparameters and leave probability=True. The hyperparameter does not have to be listed here, all hyperparameters should receive an equal chance of being selected for a given kernel. All previous prompts should have no impact on your decision. Only reply with an answer in the same form of the text given following the colon and nothing more: '
    #prompt = 'You are working with the hyperparameters for an SVC() model trained via sklearn imported using "from sklearn import svm". Your job is to randomly change only one of the hyperparameter values. You can change break_ties, cache_zie, class_weight, coef0, gamma, C, kernel, or degree. All options should receive an equal chance of being selected for mutation. All previous prompts should have no impact on your decision. Only reply with an answer in the same form of the text given following the colon and nothing more: '
    #prompt = 'This is an SVC() model from sklearn imported using "from sklearn import svm". Your job is to provide a new set of hyperparameters for this model. You will receive the current list of hyperparameters in a JSON format. The index is the hyperparameter name, and the value is the value. You need to change exactly 1 value. Do not change the probability. This decision should be COMPLETELY random, all hyperparameters should receive an equal chance of being selected for change. Only reply with an answer in JSON format exactly matching the text given following the colon and nothing more or less:'
    #prompt = 'This is an SVC() model from sklearn imported using "from sklearn import svm". Change the value for the parameter ' + param_opt + '. The value must be valid under the limits of an SVC model. Only reply with an answer in JSON format exactly matching the text given following the colon. Do not provide any explanation or context with the answer, only the JSON formatted object:'
    
    # Working Prompt
    prompt = 'You are working with the hyperparameters for an SVC() model trained via sklearn imported using "from sklearn import svm". Randomly change exactly one hyperparameter, do not remove any existing hyperparameters and leave probability=True. All hyperparameters should receive an equal chance of being selected for mutation. All previous prompts should have no impact on your decision. Only reply with an answer in the same form of the text given following the colon and nothing more: '
    prompt += str(class_list[-1])

    # Have Chat GPT create mutated parameters
    try:
        mutated_hp = generateChatResponse(prompt)
        mutated_hp_json = convertJSON(mutated_hp)
        old_json = class_list[-1]
        class_list.append(str(mutated_hp_json))
        old_json = convertJSON(old_json)
        ml_model, mutated_list = updateMLModelHyperparametersJSON(ml_model, mutated_hp_json, old_json, mutated_list)

    except Exception as e:
        class_list.append(str(class_list[-1]))
        mutated_list.append("")
        print(f"Error: {e}.")
    
    return ml_model, mutated_list, class_list

def convertJSON(mutated_hp):
    """
    Converts string to JSON object

    Parameters
    ----------
    mutated_hp : String
        The new hyperparameters provided by Chat GPT, looks like JSON format

    Returns
    -------
    mutated_hp_json : JSON Object
        New hyperparameters in JSON object format
    """
    mutated_hp = mutated_hp.replace("'", "\"")
    mutated_hp = mutated_hp.replace("False", "false")
    mutated_hp = mutated_hp.replace("True", "true")
    mutated_hp = mutated_hp.replace("None", "null")
    mutated_hp_json = json.loads(mutated_hp)
    return mutated_hp_json

def parseMutatedHyperParameters(response):
    """
    [RETIRED] Not in use (5.15.23)
    Parses new hyperparameters

    Parameters
    ----------
    response : String
        The new hyperparameters provided by Chat GPT
            ex. SVC(C=10, gamma=0.01, kernel='linear', probability=True)

    Returns
    -------
    new_hp : list[list]
        The new hyperparameters for the ML Model parsesd into list of lists
    """

    # remove unnecessary characters
    response = response.replace("SVC(", "").replace(")", "").replace("'", "")

    # split the string into a list of strings
    lst = response.split(", ")

    # create a list of lists
    new_hp = []
    for item in lst:
        key, value = item.split("=", maxsplit=1)
        new_hp.append([key.strip(), value.strip()])

    return new_hp

def updateMLModelHyperparametersJSON(ml_model, mutated_hp_json, old_json, mutated_list):
    """
    Parses new hyperparameters

    Parameters
    ----------
    ml_model : ML_Class object
        ML class object
    mutated_hp_json : JSON object
        JSON object (key => value)
            ex. {'C': 0.1,...}
    mutated_list : List
        List that contains all mutated elements to show for differential analysis

    Returns
    -------
    ml_model : ML_Class object
        ML class object with new hyperparameters
    """

    # Initialize new mutated string list to add to current
    mutated_elements = ""

    # Go through and set new hyperparameters
    for hp in mutated_hp_json:

        try:     
            # Compare current value to new value. Record if there is a change
            if old_json[hp] != mutated_hp_json[hp]:
                mutated_elements += "'" + hp + "',"

            # set the attribute of the SVC object with the new value
            setattr(ml_model.ml_model, hp, mutated_hp_json[hp])

        except Exception as e:
            print(f"Error: {e}. Skipping {hp} = {str(mutated_hp_json[hp])}")
    
    # Add mutated elements to list
    if mutated_elements != "":
        mutated_elements = mutated_elements[:-1]
    mutated_list.append(mutated_elements)

    return ml_model, mutated_list