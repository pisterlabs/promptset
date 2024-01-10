import pandas as pd

## GPT3 ##

def user_input_from_article(article) -> str:
    """
    Gets the user input from the article
    args:
        article: article dict
    returns:
        user_input: user input
    """
    user_input = f"""Title: {article['title']}\nDate: {article['date']}\nText: {article['text']}"""
    return user_input

def make_prompt(prompt: str, user_input: str) -> str:
    """
    Replaces the [user_input] in the prompt with the user input
    args:
        prompt: prompt to be used
        user_input: user input to be used
    returns:
        prompt: prompt with user input
    """
    return prompt.replace("[user_input]", user_input)

def get_rows_from_completion(completion: str, ending: str = "end") -> list[list[str]]:
    """
    Turns the text from the openai gpt response into a list of table rows
    args:
        completion: completion from openai
        ending: ending of the table (use 'end' if the table inds with '|end|')
    returns:
        list: list of table rows
    """
    # copy string
    text = completion

    # add '|' to the end of string, if it doesn't end with '|'
    if text[-1] != '|':
        text += '|'

    rows = text.split("\n")
    rows = [row.split("|") for row in rows]

    #remove trailing spaces from each row
    rows = [[item.strip() for item in row] for row in rows]
    rows = [row[1:-1] for row in rows if len(row) > 1]

    # remove ending row
    rows = [row for row in rows if row[0] != ending]

    return rows

def get_table_from_completion(completion, cols : list[str] = ['entity_1', 'relationship', 'entity_2', 'relationship_date', 'passage']) -> pd.DataFrame:
    """
    Turns the text from the openai response into a pandas dataframe
    args:
        response: response from openai
        cols: column names for the table
    returns:
        table: pandas dataframe
    """
    rows = get_rows_from_completion(completion)

    # if a row doesn't have the same number of columns as cols, then remove it
    rows = [row for row in rows if len(row) == len(cols)]

    table = pd.DataFrame(rows, columns=cols)
    return table


## ChatGPT ##
def append_user_input_to_chat(chatbot_prompt, user_input):
    """
    Appends the user input to the chatbot prompt dict
    args:
        chatbot_prompt: chatbot prompt dict
        user_input: user input
    returns:
        chatbot_prompt_cpy: chatbot prompt dict with user input appended
    """
    # create copy of chatbot prompt
    chatbot_prompt_cpy = chatbot_prompt.copy()
    prompt_dict = {'role': 'user', 'content': user_input}
    chatbot_prompt_cpy.append(prompt_dict)
    return chatbot_prompt_cpy