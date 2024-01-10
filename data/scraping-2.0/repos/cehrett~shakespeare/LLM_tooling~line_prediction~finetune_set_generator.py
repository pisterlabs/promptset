#This code scrapes all plays and creates a train, test, validation set
#It can also create the prompts for a single play by title

import llm_prompt
from csv import writer
import pandas as pd

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

def generate_csv_name(title):
    play_csv_name = title
    play_csv_name = play_csv_name.replace(' ', '_')
    play_csv_name = play_csv_name.replace(',', '')
    play_csv_name = play_csv_name.replace("'", '')
    play_csv_name = play_csv_name.replace("’", '')
    play_csv_name = play_csv_name + '_df.csv'

    return play_csv_name

def create_prompt(title, start):
    import random

    play_info_df = pd.read_csv('plays_to_scrape.csv')
    play_name_index = play_info_df[play_info_df['title']== title].index.values
    author = play_info_df['author'][play_name_index].values[0]
    year = play_info_df['year'][play_name_index].values[0]
    genre = play_info_df['genre'][play_name_index].values[0]

    #generates csv name to look for (same rules to make the original csv name)
    play_csv_name = generate_csv_name(title)
  
    play_df = pd.read_csv(play_csv_name)
    lines = ""
    
    line_index = start
    count = 0

    current_speaker = ""
    for i in range(3):
        is_dialogue = True
        if play_df.loc[line_index]['text_type'] == 'Dialogue':
            if play_df.loc[line_index]['speaker'] == current_speaker:
                lines = lines + "\t" +play_df.loc[line_index]['text'] + "\n"
            else:
                lines = lines + play_df.loc[line_index]['speaker'] + ": " + play_df.loc[line_index]['text'] + "\n "
                current_speaker = play_df.loc[line_index]['speaker']

            if count == 1:
                next_line_return = line_index
            line_index = line_index + 1 
            count += 1
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
                if count == 1:
                    next_line_return = line_index
                line_index = line_index + 1
                count += 1
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

    #this should check if index is within bounds
    #finds next line of dialogue
    if(next_line_index > play_df.tail(1).index[0]):
        next_line = ""
        next_line_index = -1
        next_line_return = -1
    else:
        while (next_line_index < play_df.tail(1).index[0]) and (play_df.loc[next_line_index]['text_type'] != 'Dialogue'):
                next_line_index = next_line_index + 1

        if play_df.loc[next_line_index]['text_type'] == 'Dialogue':
            if current_speaker != play_df.loc[next_line_index]['speaker']:
                next_line = play_df.loc[next_line_index]['speaker'] + ": " + play_df.loc[next_line_index]['text']
            else:
                next_line = play_df.loc[next_line_index]['text'] 


    if((next_line_return+2) >= play_df.tail(1).index[0]):
        next_line_return = -1

    return prompt, next_line, next_line_return

def create_test_train_val_sets():
    #create test set
    for i in range(16):
        play_info_df = pd.read_csv('plays_to_scrape.csv')
        play_name = str(play_info_df['title'][i])
        play_name_index = play_info_df[play_info_df['title']== play_name].index.values
        play_csv_name = generate_csv_name(play_name)
        # play_name_index = play_info_df[play_info_df['title']== title].index.values
        # play_csv_name = generate_csv_name(title)
        
        play_df = pd.read_csv(play_csv_name)
        line_index = (play_df['text_type'] == 'Dialogue').idxmax()

        if(line_index != -1):
            # open the file in the write mode
            with open('train_set.csv', 'a', encoding='UTF8', newline='') as f:

                while line_index != -1:
                    prompt, next_line, line_index = create_prompt(play_name, line_index)
                    # prompt, next_line, line_index = create_prompt(title, line_index)
                    # print(str(prompt))
                    # print(str(next_line))
                    #if (line_index != -1):
                    row = [str(prompt), str(next_line)]

                    # create the csv writer
                    writer_object  = writer(f)

                    # write a row to the csv file
                    writer_object.writerow(row)

    f.close()

    #creating test set
    for i in range(16, 21):
        play_name = str(play_info_df['title'][i])
        play_name_index = play_info_df[play_info_df['title']== play_name].index.values
        play_csv_name = generate_csv_name(play_name)

        
        play_df = pd.read_csv(play_csv_name)
        line_index = (play_df['text_type'] == 'Dialogue').idxmax()

        if(line_index != -1):
            # open the file in the write mode
            with open('test_set.csv', 'a', encoding='UTF8', newline='') as f:

                while line_index != -1:
                    prompt, next_line, line_index = create_prompt(play_name, line_index)
                    row = [str(prompt), str(next_line)]

                    # create the csv writer
                    writer_object  = writer(f)

                    # write a row to the csv file
                    writer_object.writerow(row)

    f.close()

    #creating validation set
    for i in range(21, 26):
        play_name = str(play_info_df['title'][i])
        play_name_index = play_info_df[play_info_df['title']== play_name].index.values
        play_csv_name = generate_csv_name(play_name)

        
        play_df = pd.read_csv(play_csv_name)
        line_index = (play_df['text_type'] == 'Dialogue').idxmax()

        if(line_index != -1):
            # open the file in the write mode
            with open('validation_set.csv', 'a', encoding='UTF8', newline='') as f:

                while line_index != -1:
                    prompt, next_line, line_index = create_prompt(play_name, line_index)
                    row = [str(prompt), str(next_line)]

                    # create the csv writer
                    writer_object  = writer(f)

                    # write a row to the csv file
                    writer_object.writerow(row)

    f.close()

def single_play_prompts(title):

    play_info_df = pd.read_csv('plays_to_scrape.csv')
    play_name = str(play_info_df['title'][i])
    play_name_index = play_info_df[play_info_df['title']== title].index.values
    play_csv_name = generate_csv_name(title)
        
    play_df = pd.read_csv(play_csv_name)
    line_index = (play_df['text_type'] == 'Dialogue').idxmax()

    if(line_index != -1):
        # open the file in the write mode
        with open('train_set.csv', 'a', encoding='UTF8', newline='') as f:

            while line_index != -1:
                prompt, next_line, line_index = create_prompt(title, line_index)
                    
                row = [str(prompt), str(next_line)]

                # create the csv writer
                writer_object  = writer(f)

                # write a row to the csv file
                writer_object.writerow(row)

create_test_train_val_sets()

# Change play that is iterated through here
# title = "Dido, Queen of Carthage"
# single_play_prompts(title)