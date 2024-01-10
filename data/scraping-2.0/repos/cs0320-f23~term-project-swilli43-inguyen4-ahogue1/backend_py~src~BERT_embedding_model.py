import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import openai
import random
import sys
import os

sys.path.append("../../..") 
sys.path.append("../..")
sys.path.append("../../../secrets")
sys.path.append("../../secrets")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../secrets")))

from API_KEY import OPENAI_API_KEY
# from secrets.API_KEY import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased',
                                    output_hidden_states = True, # Whether the model returns all hidden-states.
                                    )

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

def get_embedding(text: str):
    # Add the special tokens.
    marked_text = "[CLS] " + text + " [SEP]"

    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Mark each of the 22 tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    with torch.no_grad():

        outputs = model(tokens_tensor, segments_tensors)

        # Evaluating the model will return a different number of objects based on
        # how it's  configured in the `from_pretrained` call earlier. In this case,
        # becase we set `output_hidden_states = True`, the third item will be the
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]

        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(hidden_states, dim=0)

        token_embeddings.size()

        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        token_embeddings.size()

        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1,0,2)

        token_embeddings.size()

        # Stores the token vectors, with shape [22 x 768]
        token_vecs_sum = []

        # For each token in the sentence...
        for token in token_embeddings:

            # `token` is a [12 x 768] tensor

            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)

            # Use `sum_vec` to represent `token`.
            token_vecs_sum.append(sum_vec)

        # print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))

        # `hidden_states` has shape [13 x 1 x 22 x 768]

        # `token_vecs` is a tensor with shape [22 x 768]
        token_vecs = hidden_states[-2][0]

        # Calculate the average of all 22 token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)

        # print ("Our final sentence embedding vector of shape:", sentence_embedding.size())

        # for i, token_str in enumerate(tokenized_text):
        #     print (i, token_str)

        # print("sentence: " + text)
        # print("whole sentence vector value: " + str(sentence_embedding))
        return sentence_embedding

def calculate_similarity_between_suggestions(text1: str, text2: str):
    text1_embedding = get_embedding(text1)
    text2_embedding = get_embedding(text2)
    vector_similarity = 1 - cosine(text1_embedding, text2_embedding)
    print(f'Vector similarity for \"{text1}\" and \"{text2}\":  %.2f' % vector_similarity)
    return vector_similarity

def calculate_similarity_between_user_and_suggestion(user_embedding, suggestion: str):
    text_embedding = get_embedding(suggestion)
    vector_similarity = 1 - cosine(user_embedding, text_embedding)
    print(f'Vector similarity for user embedding and \"{suggestion}\":  %.2f' % vector_similarity)
    return vector_similarity

# def get_user_embedding(text_list):
#     new_text = ""
#     for text in text_list:
#         new_text += text + " "
#     print(new_text)
#     user_embedding = get_embedding(new_text)
#     print("user tensor1: " + str(user_embedding))
#     return user_embedding

def get_user_embedding(text_list):
    tensor_list = []
    
    for text in text_list:
        embedding = get_embedding(text)
        tensor_list.append(embedding)
    
    user_tensor = torch.stack(tensor_list, dim=0)
    user_embedding = torch.mean(user_tensor, dim=0)
    return user_embedding


def get_suggestions(user_history, user_entry: str):
    print("running main method to get personalized, ranked suggestions for user")
    """ Main function called in server endpoint
        Returns 3 perosnalized suggestions that are most similar to the user's past choices
    
        Parameters:
        - user_history List[str]: a list of the past suggestions the user has clicked on
        - user_entry (str): the user's journal entry, which is used to generate personalized
          wellbeing activity suggestions

        Returns: 
        - List[str]: the top 3 suggestions for the user
    """

    # get list of 10 activity suggestions 
    suggestion_list = generate_personalized_suggestions(user_entry)

    # calculate user embedding based on user history of suggestions clicked
    user_embedding = get_user_embedding(user_history)

    # calculate the similarity between each suggestion in suggestion_list and the user_embedding
    sorted_suggestion2similarity = rank_suggestions(user_embedding, suggestion_list)

    # return the top 3 suggestions (the top 3 most similar suggestions)
    final_suggestion_list = []
    for i in range(3):
        final_suggestion_list.append(sorted_suggestion2similarity[i][0])
    
    return final_suggestion_list

def generate_personalized_suggestions(user_entry: str):
    """ Main function to generate suggestions for user.

        Queries the LLM to generate personalized suggestions based on the user 
        entry. Confirms that LLM response is properly formatted. If it's not,
        makes an attempt to re-format the list of personalized suggestions. If
        that doesn't work, instead return 10 randomly selected suggestions from
        a pre-defined bank of suggestions.
    
        Parameters:
        - user_entry (str): to send to LLM to generate personalized 

        Returns:
        - List[str]: 10 properly formatted wellbeing suggestions for the user
    """
    
    # step 1: prompt LLM & get personalized suggestions
    raw_suggestions = prompt_LLM_for_suggestions(user_entry)

    print(f"Based on user entry, LLM returned \n{raw_suggestions}")

    if valid_format(raw_suggestions):
        print("the LLM returned suggestions in a valid format")
        return raw_suggestions
    
    # if LLM response is malformatted, attempt to reformat
    else: 
        raw_suggestions2 = reformat_LLM_suggestions(raw_suggestions)
        print("the LLM returned malformatted suggestions")
        print(f"since the LLM returned malformatted suggestions, another API request was made to \
            reformat the suggestions. the reformatted suggestions are \n{raw_suggestions2}")

        if valid_format(raw_suggestions2):
            print("attempts to reformat malformatted suggestions worked!")
            return raw_suggestions2
        
        # if LLM response is still malformatted, randmoly select 10 suggestions from 
        # a predefined bank
        else: 
            print("since the LLM failed again to return properly formatted suggestions, drawing from suggestion bank instead")
            suggestions = get_10_predefined_suggestions()
            return suggestions

def prompt_LLM_for_suggestions(user_entry: str):
    """
    Generate 10 suggestions for improving user's wellbeing based on the provided journal entry.

    Parameters:
    - user_entry (str): The journal entry from the user.

    Returns:
    - List[str]: A Python list containing 10 suggestion strings for improving wellbeing.
    """
    # openAI chat endpoint requires a list including system, assistant, and user, so we initialize that here

    # define messages array to contain instructions to generate 10 prompts based on user's journal entry
    system_content = "You are an assistant that receives journal entries and recommends wellbeing recommendations in a pythonic, un-named, comma seperated list of strings"
    LLM_instruction = f"Given this journal entry written by a user, return a comma separated list of 10 short suggestions you wold give to improve the user's wellbeing. ONLY return an un-named comma separated python list of 10 suggestions. Do not include numbers or new line chracters. \n\nEntry: {user_entry}"
    system_instruction = {"role":"system", "content": system_content}
    LLM_prompt = {"role": "user", "content": LLM_instruction}
    messages = []
    messages.append(system_instruction)
    messages.append(LLM_prompt)

    LLM_response = query_LLM(messages)

    # LLM_response = '"1. Go for a walk", "\n2. Take a nap", "\n3. Go to sleep"'
    formatted_LLM_response = LLM_response.replace('"', '') # strip all quotes
    # formatted_LLM_response = formatted_LLM_response.strip 
    # remove any numbers (e.g. list numberings)
    formatted_LLM_response = ''.join(char for char in formatted_LLM_response if not char.isnumeric())
    # remove any periods (which may linger after removing numbers)
    formatted_LLM_response = formatted_LLM_response.replace('. ', '')
    # remove any newline characters
    formatted_LLM_response = formatted_LLM_response.replace('\n', '')
    formatted_LLM_response = formatted_LLM_response.lower() # make all lower case to avoid case anomolies
    output_list = formatted_LLM_response.split(", ")

    print(f"after the first call to the LLM, the suggestions is {output_list}")

    return output_list


def reformat_LLM_suggestions(raw_suggestions):
    """
    Reformat an incorrectly formatted list of suggestions returned after initial prompt to LLM

    Parameters:
    - raw_suggestions (str?): Malformatted LLM response after initial call to prompt_LLM_for_suggestions

    Returns:
    - A reformatted response from the LLM
    """
    # define messages array w/ prompt instructing openAI to re-format prompts into a comma seperated string list
    system_content = "You are an assistant that receives a list of prompts and reformats them into a pythonic, un-named, comma seperated list of strings."
    LLM_instruction = f"Reformat the list of activities into a a pythonic, un-named, comma seperated list of strings. ONLY return the list of strings, nothing else. \n\nActivities List:{raw_suggestions}"
    
    system_instruction = {"role":"system", "content": system_content}
    LLM_prompt = {"role": "user", "content": LLM_instruction}
    messages = []
    messages.append(system_instruction)
    messages.append(LLM_prompt)

    LLM_response = query_LLM(messages) # TODO: uncomment this

    formatted_LLM_response = LLM_response.replace('"', '') # strip all quotes
    # remove any numbers (e.g. list numberings)
    formatted_LLM_response = ''.join(char for char in formatted_LLM_response if not char.isnumeric())
    # remove any periods (which may linger after removing numbers)
    formatted_LLM_response = formatted_LLM_response.replace('. ', '')
    # remove any newline characters
    formatted_LLM_response = formatted_LLM_response.replace('\n', '')
    formatted_LLM_response = formatted_LLM_response.lower() # make all lower case to avoid case anomolies
    output_list = formatted_LLM_response.split(", ")

    print(f"after querying the LLM to reformat the initial LLM response, the suggestions list is {output_list}")

    return output_list


def query_LLM(messages):
    """ Helper function that queries the LLM and parses LLM response
    """
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
        )
    # parse the LLM response
    LLM_response = completion.choices[0].message["content"]

    return LLM_response


def get_10_predefined_suggestions():
    """ Contains a pre-defined bank of 50 wellbeing suggestions from which 10 are randomly selected
        Used if the LLM does not produce properly formatted responses after 2 attempts
    """
    suggestion_bank = [
        "Practice mindfulness meditation", "Take a nature walk", "Engage in deep breathing exercises", 
        "Try yoga or tai chi", "Write in a gratitude journal", "Connect with loved ones", "Listen to uplifting music", 
        "Get a good night's sleep", "Learn a new hobby", "Limit screen time", "Eat a balanced diet", "Exercise regularly", 
        "Volunteer for a cause you care about", "Practice self-compassion", "Set realistic goals", "Unplug from technology", 
        "Read a good book", "Take short breaks throughout the day", "Express your creativity through art", 
        "Practice progressive muscle relaxation", "Attend a support group", "Establish a bedtime routine", "Laugh more - watch a comedy", 
        "Cultivate a positive mindset", "Spend time in sunlight", "Declutter your living space", "Engage in random acts of kindness",
        "Reflect on your achievements", "Learn stress management techniques", "Practice digital detox regularly"
     ]
    
    # random_suggestions is a list of 10 suggestions randomly selected from suggestion_bank w/out repetition
    random_suggestions = random.sample(suggestion_bank, 10)

    return random_suggestions

def rank_suggestions(user_embedding, suggestion_list):
    """ Ranks the suggestions for the user (based on similarity to user's past choices)

        Parameters:
        - user_embedding: a vector representing the user's past choices
        - suggestion_list List[str]: a list of 10 suggestions for the user

        Returns:
        - ranked_suggestions: a ranked list of user suggestions
    """
    suggestion2similarity = {}
    for suggestion in suggestion_list:
        suggestion_similarity_score = calculate_similarity_between_user_and_suggestion(user_embedding=user_embedding,
                                                                                       suggestion=suggestion)
        suggestion2similarity[suggestion] = suggestion_similarity_score

    # sort the suggestions in order of similarity to the user
    sorted_suggestion2similarity = sorted(suggestion2similarity.items(), key=lambda x:x[1], reverse=True)
    print(f"the sorted suggestions are: ")
    [print(x) for x in sorted_suggestion2similarity]
    return sorted_suggestion2similarity

def valid_format(LLM_suggestions): 
    """ Check if LLM_suggestions is properly formatted as a list of 10 strings

        Parameters: 
        - LLM_suggestions: a response from the LLM

        Returns: 
        - boolean: whether formatting is valid or not
    """
    # if the LLM_suggestions are not a list, format is invalid
    if not isinstance(LLM_suggestions, list):
        return False

    # if LLM_suggestions is not a list of strings, format is invalid
    for item in LLM_suggestions:
        if type(item) is not str:
            return False
    
    # if LLM_suggestions does not contain 10 suggestions, format is invalid
    if len(LLM_suggestions) != 10:
        return False
    
    # if the list is a list of 10 strings, it's format is valid
    return True

def get_mocked_suggestions():
    """ Returns a list of 10 mocked suggestions 
    """
    # query LLM w/ prompting to only return an un-named python list of strings that we can parse

    #TODO: replace mocked suggestion list with real query to an LLM that dyanmically generates 
    # a list of 10 suggestions based on entry 
    # mocked_suggestion_list = ["Practice deep breathing exercises",
    # "Try progressive muscle relaxation (PMR)",
    # "Take a walk",
    # "Participate in physical exercise",
    # "Listen to a calming music playlist",
    # "Read a book",
    # "Take short breaks to refresh your mind",
    # "Incorporate yoga for mindfulness",
    # "Spend time in nature",
    # "Enjoy a good laugh with comedy"]

    # return mocked_suggestion_list
        

# user_history = ["Take a walk outside",
#     "Go for a run",
#     "Hike a trail nearby",
#     "Try yoga or stretching",
#     "Go cycling on a path",
#     "Plan a camping trip"]
 

# print("\nfinal suggestion list:")
# mock_user_entry = "happy"
# print(get_suggestions(user_history, mock_user_entry))

# get_suggestions(user_history, mock_user_entry)


# get_embedding("Go outside for a walk.")
# calculate_similarity("Go for a walk.", "Take a run outside.")
# history = ["Go for a walk.", "Take a run outside.", "Spend time in nature."]
# user = get_user_embedding1(history)
# calculate_similarity_between_user_and_suggestion(user, "Call a friend.")
# calculate_similarity_between_user_and_suggestion(user, "Take a walk.")

# user = get_user_embedding2(history)
# calculate_similarity_between_user_and_suggestion(user, "Call a friend.")
# calculate_similarity_between_user_and_suggestion(user, "Take a walk.")