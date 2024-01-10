from langchain.prompts import PromptTemplate

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

PROMPT = """Title: {title}
Year: {year}
Author: {author}
Genre: {genre}
Characters in play: {chars}
Lines of play: 
{lines_of_play}

Next line:
"""

DEFAULT_SYSTEM_PROMPT = """You are an expert at predicting the next line of Elizabethan plays. Given the title of a play, its published year, author, genre, characters in the play, and three lines, generate the next line that follows. Only generate the next line with no other introduction or explanation."""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template


def create_prompt(title):
    import pandas as pd
    import random

    play_info_df = pd.read_csv('plays_to_scrape.csv')
    play_name_index = play_info_df[play_info_df['title']== title].index.values
    author = play_info_df['author'][play_name_index].values[0]
    year = play_info_df['year'][play_name_index].values[0]
    genre = play_info_df['genre'][play_name_index].values[0]

    #generates csv name to look for (same rules to make the original csv name)
    play_csv_name = title
    play_csv_name = play_csv_name.replace(' ', '_')
    play_csv_name = play_csv_name.replace(',', '')
    play_csv_name = play_csv_name.replace("'", '')
    play_csv_name = play_csv_name.replace("’", '')
    play_csv_name = play_csv_name + '_df.csv'
  
    play_df = pd.read_csv(play_csv_name)
    lines = ""
    
    line_index = (play_df['text_type'] == 'Dialogue').idxmax()

    current_speaker = ""
    for i in range(3):
        is_dialogue = True
        if play_df.loc[line_index]['text_type'] == 'Dialogue':
            if play_df.loc[line_index]['speaker'] == current_speaker:
                lines = lines + "\t" +play_df.loc[line_index]['text'] + "\n"
            else:
                lines = lines + play_df.loc[line_index]['speaker'] + ": " + play_df.loc[line_index]['text'] + "\n "
                current_speaker = play_df.loc[line_index]['speaker']
            line_index = line_index + 1 
        else:
            is_dialogue = False
        
        while is_dialogue == False:
            line_index = line_index + 1
            if play_df.loc[line_index]['text_type'] == 'Dialogue':
                if play_df.loc[line_index]['speaker'] == current_speaker:
                    lines = lines + "\t" +play_df.loc[line_index]['text'] + "\n"
                else:
                    lines = lines + play_df.loc[line_index]['speaker'] + ": " + play_df.loc[line_index]['text'] + "\n "
                    current_speaker = play_df.loc[line_index]['speaker']

                is_dialogue = True
                line_index = line_index + 1
            else:
                is_dialogue = False

    male_list_string = str(play_info_df['male_characters'][play_name_index].values[0])
    male_col_list = male_list_string.split(", ")

    #female_list
    female_list_string = str(play_info_df['female_characters'][play_name_index].values[0])
    female_col_list = female_list_string.split(", ")

    #other_list
    other_list_string = str(play_info_df['other_characters'][play_name_index].values[0])
    other_col_list = other_list_string.split(", ")

    #combines lists of characters and shuffles them so that it reduces bias of what character is chosen as the next speaker
    char_list = male_col_list + female_col_list + other_col_list
    random.shuffle(char_list)
    
    template = get_prompt(PROMPT)
    
    prompt = template.format(title=title, year=year, author=author, genre=genre, chars=char_list,
                        lines_of_play=lines)
    
    next_line_index = line_index
    #careful, you'll run into an error if you run out of dialogue
    while play_df.loc[next_line_index]['text_type'] != 'Dialogue':
        next_line_index = next_line_index + 1
        
    next_line = play_df.loc[next_line_index]['text']
    
    return str(prompt), next_line