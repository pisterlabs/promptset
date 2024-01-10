import json
import tkinter as tk
import numpy as np

from tkinter import messagebox, simpledialog
import os
import openai
from dotenv import load_dotenv
import shutil

from src.generate_sents import get_sentences_chatgpt
from src.azure_translate import azure_translate, augmented_azure_translate

from nltk.translate.bleu_score import sentence_bleu

data_path = 'or_en'
temp_save_path = data_path + '_temp'
sources_json_path = 'sources.json'                    # { SOURCE_ID: "" }                 # source odia word
translations_json_path = 'translations.json'          # { SOURCE_ID: "" }                 # translation of the source odia word
is_used_json_path = 'is_used.json'                    # { SOURCE_ID: True or False }      # checkbox, whether to include this translation in the augmentation 
test_sentences_json_path = 'test_sentences.json'      # { SOURCE_ID: { SUB_SENTENCE_ID: { "in": "", "out": ""} } }           # sample sentences to test on

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY_GAUTAM')
AZURE_API_KEY = os.getenv('AZURE_API_KEY_GAUTAM')
openai.api_key = OPENAI_API_KEY

try:
    shutil.rmtree(temp_save_path)
except:
    pass
shutil.copytree(data_path, temp_save_path, symlinks=False, ignore=None, copy_function=shutil.copy2, ignore_dangling_symlinks=False, dirs_exist_ok=False)



# add a button to add sentences for a word
    # the button should:
    # 1. add to respective json dictionaries
    # 2. add translations for that word using gpt3.5
    # OPTIONAL: add functionality for handling errors here later
    # 3. OPTIONAL FOR NOW: include a checkbox inside it to include the evaluation for that word in the final or not.
# add a button to remove a selected translation from the dictionary
# add checkboxes for each word to check ON or OFF for being used in the translation
# add a button to hide words not checked ON
# add a button to evaluate BLEU score
# OPTIONAL: add a button to add translations manually
# DO LATER: add button to overwrite changes in the original directory

# Note: everytime any changes are made to the dictionary, changes have to be synced everywhere else using:
            # 1. save_translations(translations)
            # 2. update_listbox()

def load_translations(parent_folder): # this thing has to be tight. so an error will be thrown if some unexpected behaviour happens.
    paths = [os.path.join(parent_folder, filepath)
            for filepath 
            in [sources_json_path, translations_json_path, is_used_json_path, test_sentences_json_path]]
    files = []
    for path in paths:
        try:
            with open(path, 'r', encoding='utf-8') as file:
                files.append({int(key):value for key, value in json.load(file).items()})
        except Exception as e:
            print('error while reading ', path)
            raise e # no exceptions allowed. file reading has to be tight for now.
    sources, translations, is_used, test_sentences = files
    # print(type(list(is_used.values())[0]))
    return sources, translations, is_used, test_sentences

def save_sources(parent_folder, sources):
    with open(os.path.join(parent_folder, sources_json_path), 'w', encoding='utf-8') as file:
        json.dump(sources, file, indent=4)

def save_translations(parent_folder, translations):
    with open(os.path.join(parent_folder, translations_json_path), 'w', encoding='utf-8') as file:
        json.dump(translations, file, indent=4)

def save_is_used(parent_folder, is_used):
    with open(os.path.join(parent_folder, is_used_json_path), 'w', encoding='utf-8') as file:
        json.dump(is_used, file, indent=4)

def save_test_sentences(parent_folder, test_sentences):
    with open(os.path.join(parent_folder, test_sentences_json_path), 'w', encoding='utf-8') as file:
        json.dump(test_sentences, file, indent=4)

def save_all(parent_folder):
    save_sources(parent_folder, sources)
    save_translations(parent_folder, translations)
    save_is_used(parent_folder, is_used)
    save_test_sentences(parent_folder, test_sentences)
    if parent_folder == data_path:
        messagebox.showinfo("SUCCESS", f"Successfully saved to original source folder.")

def add_translation(SOURCE_ID, source, translation, checkbox, parent_folder):
    # assign to dicts
    sources[SOURCE_ID] = source
    translations[SOURCE_ID] = translation
    is_used[SOURCE_ID] = checkbox
    # print(type(checkbox))

    # generate sentences using GPT3.5 and assign to dicts
    generated_sents = get_sentences_chatgpt(source, translation)
    test_sentences[SOURCE_ID] = { 
        i: {
            'in': sent_source, 
            'out': sent_translation
        } 
        for i, (sent_source, sent_translation) 
        in enumerate(generated_sents)
    }

    # call save_translations, save_sources, etc.
    save_all(parent_folder)

    # update the view inside the GUI
    #DEBUGGING
    entries = my_game.get_children()
    entry_count = len(entries)
    my_game.insert(parent='', index='end', iid=entry_count, text='', values=(SOURCE_ID, source, translation, checkbox))

def clear_entry():
    # clear entry boxes
    entry0.delete(0, tk.END)
    entry1.delete(0, tk.END)
    entry2.delete(0, tk.END)
    entry3.delete(0, tk.END)

def update_entry(event):
    selected_item = my_game.focus()
    clear_entry()
    if selected_item:
        values = my_game.item(selected_item)['values']  # Get the values of the selected item
        entry0.insert(tk.END, values[0])
        entry1.insert(tk.END, values[1])
        entry2.insert(tk.END, values[2])
        entry3.insert(tk.END, values[3])

def _debug_entry():
    print([entry0.get(), entry1.get(), entry2.get(), entry3.get()])

def generate_new_source_id():
    return (0 if not len(sources) else max(sources.keys())) + 1

def new_entry():
    SOURCE_ID = generate_new_source_id()
    entry0.delete(0, tk.END)
    entry0.insert(tk.END, SOURCE_ID)

def add_update_record():
    # select the record based on the source id given
    try:
        SOURCE_ID = int(entry0.get())
    except: # if SOURCE_ID == ''
        SOURCE_ID = generate_new_source_id()
    if entry1 == '' or entry2 == '' or entry3 == '': # no insertion of empty values
        messagebox.showinfo("Error", "ERROR: you tried inserting empty values.")
        return
    
    # save new data
    add_translation(SOURCE_ID, entry1.get(), entry2.get(), True if entry3.get() == 'True' else False, temp_save_path)

    # clear entry boxes
    clear_entry()

def remove_translation():
    try:
        SOURCE_ID = int(entry0.get())
        sources[SOURCE_ID]
    except:
        messagebox.showinfo("Error", f"ERROR: No record exists with the Source_Id: {entry0.get()}")
        return
    for item in my_game.get_children():
        values = my_game.item(item, 'values')
        print(values)
        if int(values[0]) == SOURCE_ID:
            my_game.delete(item)
            del sources[SOURCE_ID], translations[SOURCE_ID], is_used[SOURCE_ID], test_sentences[SOURCE_ID]
            messagebox.showinfo("SUCCESS", f"Successfully deleted record with Source_Id: {entry0.get()}")
    
def score_translations():
    data_dict = {sources[SOURCE_ID]: translations[SOURCE_ID] for SOURCE_ID in sources.keys() if is_used[SOURCE_ID]}
    bleu_scores = []
    for SOURCE_ID in sources.keys():
        for SENTENCE_ID in test_sentences[SOURCE_ID].keys():
            source_sent = test_sentences[SOURCE_ID][SENTENCE_ID]['in']
            target_sent = test_sentences[SOURCE_ID][SENTENCE_ID]['out']
            translated_sent = augmented_azure_translate('en', source_sent, data_dict)

            # Convert the sentences to lists of tokens
            reference_tokens = target_sent.split()
            candidate_tokens = translated_sent.split()

            # Calculate the BLEU score
            bleu_score = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25))
            bleu_scores.append(bleu_score)
    score = np.mean(bleu_scores)
    evaluate_entry.delete(0, tk.END)
    evaluate_entry.insert(tk.END, score)



    


sources, translations, is_used, test_sentences = load_translations(data_path)

root = tk.Tk()
root.title("Translation POC - Editor")
root.geometry("800x800")
root["bg"] = "#AC99F2"

game_frame = tk.Frame()
game_frame.pack()

# scrollbar
game_scroll = tk.Scrollbar(game_frame)
game_scroll.pack(side=tk.RIGHT, fill=tk.Y)

game_scroll = tk.Scrollbar(game_frame,orient='horizontal')
game_scroll.pack(side=tk.BOTTOM, fill=tk.X)

from tkinter import ttk
my_game = ttk.Treeview(game_frame,yscrollcommand=game_scroll.set, xscrollcommand =game_scroll.set)
my_game.pack()

game_scroll.config(command=my_game.yview)
game_scroll.config(command=my_game.xview)

# define our column
my_game['columns'] = ('id', 'source', 'translation', 'is used')

# format our column
my_game.column("#0", width=0, stretch=tk.NO)
my_game.column(my_game['columns'][0], anchor=tk.CENTER, width=100)
my_game.column(my_game['columns'][1], anchor=tk.CENTER, width=250)
my_game.column(my_game['columns'][2], anchor=tk.CENTER, width=250)
my_game.column(my_game['columns'][3], anchor=tk.CENTER, width=150)

# Create Headings 
my_game.heading("#0", text="", anchor=tk.CENTER)
my_game.heading(my_game['columns'][0], text="Id",   anchor=tk.CENTER)
my_game.heading(my_game['columns'][1], text="Source",   anchor=tk.CENTER)
my_game.heading(my_game['columns'][2], text="Translation", anchor=tk.CENTER)
my_game.heading(my_game['columns'][3], text="Used in Augmentation", anchor=tk.CENTER)

# add data from the dicts
for source_id in sources.keys():
    # print(type(is_used[source_id]))
    my_game.insert('', 'end', text="", values=(source_id, 
                                               sources[source_id], 
                                               translations[source_id],
                                               is_used[source_id]))

# define entry frame
frame = tk.Frame(root, width=800, height=50)
frame.pack(padx=10, pady=10)

# Labels for frame
label0 = tk.Label(frame, text="Id")
label1 = tk.Label(frame, text="Source")
label2 = tk.Label(frame, text="Translation")
label3 = tk.Label(frame, text="Used in Augmentation")
label0.grid(row=0, column=0)
label1.grid(row=0, column=1)
label2.grid(row=0, column=2)
label3.grid(row=0, column=3)

# Define entry boxes for frame
entry0 = tk.Entry(frame)
entry1 = tk.Entry(frame)
entry2 = tk.Entry(frame)
entry3 = tk.Entry(frame)
entry0.grid(row=1, column=0, sticky='nsew', padx=2)
entry1.grid(row=1, column=1, sticky='nsew', padx=2)
entry2.grid(row=1, column=2, sticky='nsew', padx=2)
entry3.grid(row=1, column=3, sticky='nsew', padx=2)

# Set column widths for frame
frame.columnconfigure(0, weight=100)
frame.columnconfigure(1, weight=250)
frame.columnconfigure(2, weight=250)
frame.columnconfigure(3, weight=150)

# update button to update values
# 1. make sure that the update fucnction assigns a new value if the source id field is not selected

# Bind the selection event to the update_entry function
my_game.bind('<<TreeviewSelect>>', update_entry)

edit_button = tk.Button(root, text="Add/Edit Translation", command=add_update_record)
edit_button.pack(pady = 10)

clear_button = tk.Button(root, text="Clear Entry Box", command=clear_entry)
clear_button.pack(pady = 10)

new_button = tk.Button(root, text="Generate new Id", command=new_entry)
new_button.pack(pady = 10)

remove_button = tk.Button(root, text="Remove Translation", command=remove_translation)
remove_button.pack(pady=10)

save_button = tk.Button(root, text="Save (overwrites original file with the temp. save)", command=lambda:save_all(data_path))
save_button.pack(pady=10)

score_button = tk.Button(root, text="Score Translations", command=score_translations)
score_button.pack(pady=10)


# define BLEU score frame
frame = tk.Frame(root, width=200, height=50)
frame.pack(padx=10, pady=10)
evaluate_label = tk.Label(frame, text='BLEU score:', font=('Arial', 18))
evaluate_label.grid(row=0, column=0)
evaluate_entry = tk.Entry(frame, font=('Arial', 18))
evaluate_entry.grid(row=0, column=1)


root.mainloop()

# part 2
# 200 sentences for testing / scoring
# call api for re-scoring only the sentences which get affected by an added / removed translation

# part 3
# add a button to generate_tests for like 5 sentences for the selected translation

# part 4 
# keep a general button to translate on all the sentences and calculate the BLEU score
# decouple it from ADD and REMOVE

# part 5
# rename variables in the code