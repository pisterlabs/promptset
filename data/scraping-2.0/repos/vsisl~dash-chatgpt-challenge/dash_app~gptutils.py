"""
Functions making openAI API calls.
"""
import os
import ast
import json
import openai  # chat-gpt API
from dash_app.flask_cache import cache

openai.api_key = os.getenv("OPENAI_API_KEY")


@cache.memoize(timeout=7 * 24 * 3600)
def get_completion(prompt, model="gpt-3.5-turbo"):
    """Creates chatGPT response for a given prompt.

    :param prompt: str; prompt for chat GPT
    :param model: str; optional, default: "gpt-3.5-turbo"
    :return: str: response
             int: total API call tokens used by this function call
    """
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=300,
        temperature=0.3,  # this is the degree of randomness of the model's output, bigger number = greater 'creativity'
    )
    return response.choices[0].message["content"], response["usage"]["total_tokens"]


# def get_image(text):
#     """Creates an image based on the input text
#
#     :param prompt: str
#     :param model: str; optional, default: "gpt-3.5-turbo"
#     :return: str: response
#     """
#     # input_img = Image.open("moscow.png").convert('RGBA')
#     # mask_img = Image.open("moscow_mask.png").convert('RGBA')
#     # import pdb; pdb.set_trace()
#     # import pdb; pdb.set_trace()
#     response = openai.Image.create_edit(
#         image=open("moskau.png", "rb"),
#         mask=open("moskau_mask.png", "rb"),
#         prompt="A city surrounded by {text} in pastel colors",
#         n=1,
#         size="256x256",
#     )
#
#     image_url = response["data"][0]["url"]
#     print(image_url)
#     import pdb
#
#     pdb.set_trace()
#     response = openai.Image.create(prompt=prompt, n=1, size="256x256")
#     return response.choices[0].message["content"], response["usage"]["total_tokens"]


def get_classification(input_text, model="text-davinci-003"):
    """Takes given text and classifies individual sentences for presence of propaganda techniques.

    :param input_text: str
    :param model: str; optional, default "text-davinci-003"
    :return: str
    """
    response = openai.Completion.create(
        model=model,
        # prompt=f'The following is a list of techniques used in propaganda texts: 'Appeal to Authority', 'Appeal to Fear Prejudice', 'Bandwagon or Reductio ad hitlerum', 'Black and White Fallacy', 'Causal Oversimplification', 'Doubt', 'Exaggeration or Minimisation', 'Flag-Waving',  'Loaded Language', 'Name Calling or Labeling', 'Repetition', 'Slogans', 'Thought-terminating Cliches', 'Whataboutism or Straw Man or Red Herring'.\n\nThe following is a list of techniques and their defintions used in propaganda texts: \n'Appeal to Authority' - Stating that a claim is true simply because a valid authority or expert on the issue said it was true, without any other supporting evidence offered.\n'Appeal to Fear Prejudice' - Seeking to build support for an idea by instilling anxiety and/or panic in the population towards an alternative.\n'Bandwagon or Reductio ad hitlerum' - Attempting to persuade the target audience to join in and take the course of action because "everyone else is taking the same action".\n'Black and White Fallacy'-Presenting two alternative options as the only possibilities, when in fact more possibilities exist. As an the extreme case, tell the audience exactly what actions to take, eliminating any other possible choices.\n'Causal Oversimplification'-Assuming a single cause or reason when there are actually multiple causes for an issue.\n'Doubt'-Questioning the credibility of someone or something.\n'Exaggeration or Minimisation'-Either representing something in an excessive manner: making things larger, better, worse (e.g., "the best of the best", "quality guaranteed") or making something seem less important or smaller than it really is (e.g., saying that an insult was just a joke). \n'Flag-Waving'- Playing on strong national feeling (or to any group; e.g., race, gender, political preference) to justify or promote an action or idea \n'Loaded Language'- Using specific words and phrases with strong emotional implications (either positive or negative) to influence an audience.\n'Name Calling or Labeling'-Labeling the object of the propaganda campaign as either something the target audience fears, hates, finds undesirable or loves, praises. \n'Repetition'-Repeating the same message over and over again so that the audience will eventually accept it.\n'Slogans'-A brief and striking phrase that may include labeling and stereotyping. Slogans tend to act as emotional appeals.\n'Thought-terminating Cliches'-Words or phrases that discourage critical thought and meaningful discussion about a given topic. They are typically short, generic sentences that offer seemingly simple answers to complex questions or that distract attention away from other lines of thought. \n'Whataboutism or Straw Man or Red Herring'- A technique that attempts to discredit an opponent\'s position by charging them with hypocrisy without directly disproving their argument, when an opponent\'s proposition is substituted with a similar one which is then refuted in place of the original proposition or introducing irrelevant material to the issue being discussed, so that everyone\'s attention is diverted away from the points made.\n\nIn the following text identify which of these techniques are present, it can be multiple or none. Enclose the found techniques in square brackets imitating input for Python list. Separate the techniques with a comma, put parantheses around each found technique, e.g. ['Flag-Waving',  'Loaded Language'] or None if no technique is found.\n\nText: {input_text}\nFound techniques:',
        prompt=f"The following is a list of techniques used in propaganda texts: \
        'Appeal to Authority', 'Appeal to Fear Prejudice', 'Bandwagon or Reductio ad hitlerum', \
        'Black and White Fallacy', 'Causal Oversimplification', 'Doubt', 'Exaggeration or Minimisation', 'Flag-Waving',\
        'Loaded Language', 'Name Calling or Labeling', 'Repetition', 'Slogans', 'Thought-terminating Cliches',\
        'Whataboutism or Straw Man or Red Herring'.The following is a list of techniques and their definitions used in \
        propaganda texts: 'Appeal to Authority' - Stating that a claim is true simply because a valid authority or \
        expert on the issue said it was true, without any other supporting evidence offered.'Appeal to Fear Prejudice' \
        - Seeking to build support for an idea by instilling anxiety and/or panic in the population towards an \
        alternative.'Bandwagon or Reductio ad hitlerum' - Attempting to persuade the target audience to join in \
        and take the course of action because 'everyone else is taking the same action'.'Black and White Fallacy'-\
        Presenting two alternative options as the only possibilities, when in fact more possibilities exist. As an the\
        extreme case, tell the audience exactly what actions to take, eliminating any other possible choices.'Causal\
        Oversimplification'-Assuming a single cause or reason when there are actually multiple causes for an issue.\
        'Doubt'-Questioning the credibility of someone or something.'Exaggeration or Minimisation'-Either\
        representing something in an excessive manner: making things larger, better, worse (e.g., 'the best of the \
        best', 'quality guaranteed') or making something seem less important or smaller than it really is (e.g., \
        saying that an insult was just a joke). 'Flag-Waving'- Playing on strong national feeling (or to any group; \
        e.g., race, gender, political preference) to justify or promote an action or idea 'Loaded Language'- Using \
        specific words and phrases with strong emotional implications (either positive or negative) to influence an \
        audience.'Name Calling or Labeling'-Labeling the object of the propaganda campaign as either something the \
        target audience fears, hates, finds undesirable or loves, praises. 'Repetition'-Repeating the same message \
        over and over again so that the audience will eventually accept it.'Slogans'-A brief and striking phrase that \
        may include labeling and stereotyping. Slogans tend to act as emotional appeals.'Thought-terminating Cliches'-\
        Words or phrases that discourage critical thought and meaningful discussion about a given topic. They are \
        typically short, generic sentences that offer seemingly simple answers to complex questions or that distract \
        attention away from other lines of thought. 'Whataboutism or Straw Man or Red Herring'- A technique that \
        attempts to discredit an opponent's position by charging them with hypocrisy without directly disproving \
        their argument, when an opponent's proposition is substituted with a similar one which is then refuted in \
        place of the original proposition or introducing irrelevant material to the issue being discussed, so that \
        everyone's attention is diverted away from the points made.In the following text presented after the \
        word 'Text' identify which of these techniques are present, it can be multiple or none. Enclose the found \
        techniques in square brackets imitating input for Python list consisting of strings. Separate the techniques \
        with a comma, put parantheses around each found technique, e.g. ['Flag-Waving',  'Loaded Language'] or None if \
        no technique is found.Text: {input_text}\n"
        + "\n\n###\n\n",
        temperature=0,
        max_tokens=35,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"],
    )

    return response["choices"][0]["text"]


def get_classification_cheaper(sentence, model="gpt-3.5-turbo"):
    """Takes given sentence and classifies it based on presence of propaganda techniques.

    Uses chatGPT as a classifier.

    :param sentence: str; a sentence; e.g. "BREAKING NEWS: Russian Propaganda Exposed!"
    :param model: str; optional, default: "gpt-3.5-turbo"
    :return: list of str: list of propaganda techniques discovered in the given sentence;
                            e.g. ['Loaded Language']
             int: total API call tokens used by this function call
    """
    messages = [
        {
            "role": "system",
            "content": "You are a model that finds propaganda techniques in a provided text. The following is a list of techniques and their definitions used in propaganda texts:\n\n'Appeal to Authority' - Stating that a claim is true simply because a valid authority or expert on the issue said it was true, without any other supporting evidence offered.\n\n'Appeal to Fear Prejudice' - Seeking to build support for an idea by instilling anxiety and/or panic in the population towards an alternative.\n\n'Bandwagon or Reductio ad hitlerum' - Attempting to persuade the target audience to join in and take the course of action because everyone else is taking the same action.\n\n'Black and White Fallacy' - Presenting two alternative options as the only possibilities, when in fact more possibilities exist. As an extreme case, tell the audience exactly what actions to take, eliminating any other possible choices.\n\n'Causal Oversimplification' - Assuming a single cause or reason when there are actually multiple causes for an issue.\n\n'Doubt' - Questioning the credibility of someone or something.\n\n'Exaggeration or Minimisation' - Either representing something in an excessive manner: making things larger, better, worse (e.g., the best of the best, quality guaranteed) or making something seem less important or smaller than it really is (e.g., saying that an insult was just a joke).\n\n'Flag-Waving' - Playing on strong national feeling (or to any group; e.g., race, gender, political preference) to justify or promote an action or idea.\n\n'Loaded Language' - Using specific words and phrases with strong emotional implications (either positive or negative) to influence an audience.\n\n'Name Calling or Labeling' - Labeling the object of the propaganda campaign as either something the target audience fears, hates, finds undesirable or loves, praises.\n\n'Repetition' - Repeating the same message over and over again so that the audience will eventually accept it.\n\n'Slogans' - A brief and striking phrase that may include labeling and stereotyping. Slogans tend to act as emotional appeals.\n\n'Thought-terminating Cliches' - Words or phrases that discourage critical thought and meaningful discussion about a given topic. They are typically short, generic sentences that offer seemingly simple answers to complex questions or that distract attention away from other lines of thought.\n\n'Whataboutism or Straw Man or Red Herring' - A technique that attempts to discredit an opponent's position by charging them with hypocrisy without directly disproving their argument, when an opponent's proposition is substituted with a similar one which is then refuted in place of the original proposition or introducing irrelevant material to the issue being discussed, so that everyone's attention is diverted away from the points made.",
        },
        {
            "role": "user",
            "content": "In the following text identify which of these techniques are present in a sentence, it can be multiple or none. Enclose the found techniques in each sentence in square brackets imitating input for Python list. Separate the techniques with a comma, put parentheses around each found technique, e.g. ['Flag-Waving',  'Loaded Language'] or None if no technique is found.",
        },
        {
            "role": "assistant",
            "content": "Okay, I understand. I will either return a list containing the found techniques or None. For multiple sentences a will return a combination of lists and None delimited by commas.",
        },
        {
            "role": "user",
            "content": "The global economy would plunge into chaos, with skyrocketing fuel prices, widespread unemployment, and a devastating blow to industries heavily reliant on oil.",
        },
        {"role": "assistant", "content": "['Appeal to Fear Prejudice']"},
        {
            "role": "user",
            "content": "Democrats bolted as soon as Trump's speech ended in an apparent effort to signal they can't even stomach being in the same room as the president.",
        },
        {
            "role": "assistant",
            "content": "['Exaggeration or Minimisation', 'Loaded Language']",
        },
        {
            "role": "user",
            "content": "By banning oil, they aim to cripple not only their adversaries but also their allies, leaving them at the mercy of their own agenda.",
        },
        {
            "role": "assistant",
            "content": "['Exaggeration or Minimisation']",
        },
        {
            "role": "user",
            "content": "Together, we can expose America's true intentions and safeguard the world from their manipulative grasp.",
        },
        {
            "role": "assistant",
            "content": "['Appeal to Fear Prejudice']",
        },
        {
            "role": "user",
            "content": "This is a car.",
        },
        {
            "role": "assistant",
            "content": "None",
        },
        {
            "role": "user",
            "content": "Our vast reserves and unwavering determination will ensure that the global oil market remains stable and prosperous.",
        },
        {
            "role": "assistant",
            "content": "['Loaded Language']",
        },
        {"role": "user", "content": sentence},
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )

    # de-stringify stringified list
    propaganda_techniques = ast.literal_eval(response.choices[0].message["content"])
    # TODO: try to avoid using literal_eval. I tried to use json.loads() instead of ast.literal_eval() but it didn't
    #  work.

    tokens_used = response["usage"]["total_tokens"]

    return propaganda_techniques, tokens_used


@cache.memoize(timeout=7 * 24 * 3600)
def get_classification_christian(sentence, model="gpt-3.5-turbo"):
    """Takes given sentence and classifies it based on presence of propaganda techniques.

    Uses chatGPT as a classifier.

    :param sentence: str; a sentence; e.g. "BREAKING NEWS: Russian Propaganda Exposed!"
    :param model: str; optional, default: "gpt-3.5-turbo"
    :return: response: dict; contains the following key-value pairs:
                                'classes': list of str; list of propaganda techniques
                                'confidence': list of numerical values between 0 and 1; confidence score for each
                                              propaganda technique
                                'explain': list of str; explanations why given propaganda techniques were assigned
             used_tokens int: total API call tokens used by this function call
    """
    prompt = f""" classify the sentence delimited by triple backticks into the following list of classes:
                         "Appeal to Authority", "Appeal to Fear Prejudice", "Bandwagon, Reductio ad hitlerum", "Black and White Fallacy", \
                         "Causal Oversimplification", "Doubt", "Exaggeration, Minimisation", "Flag-Waving", "Loaded Language", \
                         "Name Calling, Labeling", "Repetition", "Slogans", "Thought-terminating Cliches", \
                         "Whataboutism, Straw Man, Red Herring" \
                         As an output, give me a python dictionary with the following keys:
                         1. "classes" where all the classes are saved as a list. Use only the classes provided at the beginning with the exact spelling. If no class was classified, instead of a list return None.
                         2. "confidence" where a numerical value is assigned to each class representing the confidence of how the class is present in the text. 1 is the maximum number, 0 the lowest. \
                             Just give me a list as output where the order corresponds to the order of the classes. If no class was classified, instead of a list return None. \
                         3. "explain" where an explanation is given why you have classified the sentence with this class. Don't repeat the sentence and keep it concise. If no class was classified, instead of a list return None.
                          ```{sentence}````
               """
    message = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=message,
        temperature=0,  # this is the degree of randomness of the model's output
    )

    # extract data of interest
    try:
        # try converting stringified dictionary into a dict using json
        classification_result = json.loads(response.choices[0].message["content"])
    except json.decoder.JSONDecodeError:
        # if json fails try literal eval
        classification_result = ast.literal_eval(response.choices[0].message["content"])
    except:
        pass
        # TODO: make sure the function always returns something reasonable

    used_tokens = response["usage"]["total_tokens"]

    # validate that the response is a dictionary with the expected keys
    if type(classification_result) is not dict:
        raise TypeError(f"Response is not a dictionary.")

    # check that the dictionary returned by GPT has expected keys
    if not any(
        key in classification_result.keys()
        for key in ["classes", "confidence", "explain"]
    ):
        raise ValueError(f"Response does not contain the expected keys.")

    # make sure that all propaganda techniques identified by the classifier are among the expected options
    if classification_result["classes"] is not None:
        # Note: We had a case when the value of classification_result['classes'] was ['Appeal to Authority', 'None']
        #  the 'None' value was not supposed to be there and caused the function entity() to crash...
        #  If technique is not found in the list of techniques, we need to remove also the score and explaination.

        for i, technique in enumerate(classification_result["classes"]):
            # TODO: store a list of all supported techniques in some separate module
            if technique not in [
                "Appeal to Authority",
                "Appeal to Fear Prejudice",
                "Bandwagon, Reductio ad hitlerum",
                "Black and White Fallacy",
                "Causal Oversimplification",
                "Doubt",
                "Exaggeration, Minimisation",
                "Flag-Waving",
                "Loaded Language",
                "Name Calling, Labeling",
                "Repetition",
                "Slogans",
                "Thought-terminating Cliches",
                "Whataboutism, Straw Man, Red Herring",
            ]:
                # if technique is not from the list of supported options, remove it
                classification_result["classes"].pop(i)
                classification_result["confidence"].pop(i)
                classification_result["explain"].pop(i)

                # if there are no techniques left, assign None values (instead of empty lists)
                if len(classification_result["classes"]) == 0:
                    classification_result["classes"] = None
                    classification_result["confidence"] = None
                    classification_result["explain"] = None

    # TODO: implement further data validation

    return classification_result, used_tokens
