import os
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer, T5ForConditionalGeneration, T5Tokenizer
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import *
from pathlib import Path
from pygame import mixer
import re
import torch
import openai
import json
import time

# set the parent directory to the directory of this file
parent_dir = Path(__file__).parent
mixer.init()
mixer.music.load(Path(__file__).with_name('xuehuapiaopiao.mp3'))
mixer.music.play(-1)

# Read the text file and get the list of models(model path)
def get_model():
    try:
        with open(Path(__file__).with_name('models.txt'), 'r') as f:
            modl = f.readlines()
            modl = [i.strip() for i in modl]
            modl = [i.split(', ') for i in modl]
            modl = {i[0]: i[1] for i in modl}
            # list of model paths
            modl_name = list(modl.keys())
            modl_dir = list(modl.values())
    except:
        # search for the model paths in the parent directory
        modl_name = []
        modl_dir = []
        translators = parent_dir / 'translators'
        for i in os.listdir(translators):
            if 'mmodel' in i or 'mcheckpoint' in i or 't5model' in i or 't5checkpoint' in i or 'gpt' in i or 'hanvie' in i:
                modl_name.append(i)
                modl_dir.append(translators / i)
        # if the text file doesn't exist, create one
        with open(Path(__file__).with_name('models.txt'), 'w') as f:
            for i in range(len(modl_name)):
                f.write(modl_name[i] + ', ' + str(modl_dir[i]) + '\n')
        return get_model()

    return modl_name, modl_dir

def get_lang():
    try:
        with open(Path(__file__).with_name('lang.txt'), 'r') as f:
            # list of languages
            lang = f.readlines()
            lang = [i.strip() for i in lang]
            print(lang)
    except:
        with open(Path(__file__).with_name('lang.txt'), 'w') as f:
            f.write('Chinese')
            f.write('\n')
            f.write('Vietnamese')
            f.write('\n')
            f.write('English')
        return get_lang()

    return lang 

modl_name, modl_dir = get_model()
model_name = modl_name[-1]
model_dir = modl_dir[-1]
print(model_name, model_dir)
language = get_lang()[-1]
print(language)

# if the current model path has 'model-finetuned' in it, use the MarianMTModel class
if 'mmodel' in model_dir or 'mcheckpoint' in model_dir:
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianMTModel.from_pretrained(model_dir)
# if the current model path has 't5-small' in it, use the T5ForConditionalGeneration class
elif 't5model' in model_dir or 't5checkpoint' in model_dir:
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
# if the current model path has 'gpt' in it, use the gpt 3.5 turbo class
elif 'gpt' in model_dir:
    openai.api_key = "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"
    # get model id from txt file inside the model folder
    with open(model_dir + '/model_id.txt', 'r') as f:
        model_id = f.read()
    model = model_id
    print(model)
    tokenizer = ''
elif 'hanvie' in model_dir:
    modeld = model_dir + '/model'
    print(modeld)
    tokenizer = T5Tokenizer.from_pretrained(modeld)
    model = T5ForConditionalGeneration.from_pretrained(modeld)
    print(model)

def punctuation_process(punctuation):
    punctuation = ['. ' if i == '。' else i for i in punctuation]
    punctuation = ['! ' if i == '！' else i for i in punctuation]
    punctuation = ['? ' if i == '？' else i for i in punctuation]
    punctuation = ['... ' if i == '……' else i for i in punctuation]
    punctuation = ['?? ' if i == '？？' else i for i in punctuation]
    punctuation = ['!?! ' if i == '！？！' else i for i in punctuation]
    punctuation = ['!! ' if i == '！！' else i for i in punctuation]
    punctuation = ['?! ' if i == '？！' else i for i in punctuation]
    return punctuation

def translate_s(text):
    global model_name, model_dir, tokenizer, model
    model_switch()
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True), max_length=510)
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    ouch = tgt_text[0]
    return ouch

def translate_hugging(text):
    # split the text to many sentences based on the punctuation marks: '。','！','？', '……' at the end of each sentence
    sentences = re.split(r'[。！？……\n]', text)
    # remove the empty strings and any new line in the list
    sentences = [i.strip() for i in sentences if i != '']
    punctuation = re.findall(r'([。！？……\n]+)', text)
    punctuation = punctuation_process(punctuation)

    # translate each sentence and save them to a list
    out = [translate_s(i) for i in sentences]
    print(out)
    # join the list of sentences with the exact punctuation marks back to the end of each sentence
    if punctuation == []:
        out = [i for i in out]
    else:
        print(out)
        print(punctuation)
        out = [i + punctuation[j] for j, i in enumerate(out)]
    # join the list of sentences to a string
    out = ''.join(out)

    output_box.configure(state='normal')
    output_box.delete('1.0', 'end')
    output_box.insert('1.0', out)
    output_box.configure(state='disabled')
    return out

def translate_gpt(text, model):
    system_message = 'Translate the following Chinese text to Vietnamese, ensuring that game mechanics, character names, and plot elements are accurately conveyed.'
    test_messages = []
    test_messages.append({"role": "system", "content": system_message})
    user_message = text
    test_messages.append({"role": "user", "content": user_message})
    response = openai.ChatCompletion.create(model=model, messages=test_messages, temperature=0, max_tokens=200)
    out = response["choices"][0]["message"]["content"]
    out = out.replace('"', '')

    output_box.configure(state='normal')
    output_box.delete('1.0', 'end')
    output_box.insert('1.0', out)
    output_box.configure(state='disabled')
    return out

def translate_hanvie(text):
    map_path = model_dir + '/map.json'
    with open(map_path, 'r', encoding='utf-8') as f:
        map = json.load(f)
    print(map)
    for i in text:
        try:
            x = ' ' + map[i] + ' '
            text = text.replace(i, x)
        except:
            continue

    nugget = text.strip()
    print(nugget)
    out = translate_s(nugget)
    print(out)

    output_box.configure(state='normal')
    output_box.delete('1.0', 'end')
    output_box.insert('1.0', out)
    output_box.configure(state='disabled')
    return out

def translate(text):
    if 'gpt' in model_dir:
        translate_gpt(text, model)
    elif 'hanvie' in model_dir:
        translate_hanvie(text)
    else:
        translate_hugging(text)

def model_switch():
    global model_name, model_dir, tokenizer, model
    # get the current model path
    modl_name, modl_dir = get_model()
    # if the current model path is the same as the previous one, skip the model assignment
    if modl_dir[-1] != model_dir:
        model_name = modl_name[-1]
        model_dir = modl_dir[-1]
        # if the current model path has 'model-finetuned' or 'checkpoint' in it, use the MarianMTModel class
        if 'mmodel' in model_dir or 'mcheckpoint' in model_dir:
            tokenizer = MarianTokenizer.from_pretrained(model_dir)
            model = MarianMTModel.from_pretrained(model_dir)
        # if the current model path has 't5-small' in it, use the T5ForConditionalGeneration class
        elif 't5model' in model_dir or 't5checkpoint' in model_dir:
            tokenizer = T5Tokenizer.from_pretrained(model_dir)
            model = T5ForConditionalGeneration.from_pretrained(model_dir)
        # if the current model path has 'gpt' in it, use the gpt 3.5 turbo class
        elif 'gpt' in model_dir:
            openai.api_key = "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"
            # get model id from txt file inside the model folder
            with open(model_dir + '/model_id.txt', 'r') as f:
                model_id = f.read()
            model = model_id
            tokenizer = ''
        elif 'hanvie' in model_dir:
            modeld = model_dir + '/model'
            tokenizer = T5Tokenizer.from_pretrained(modeld)
            model = T5ForConditionalGeneration.from_pretrained(modeld)
    return model, tokenizer

def language_switch():
    if language == 'English':
        eng_config()
    elif language == 'Vietnamese':
        viet_config()
    elif language == 'Chinese':
        china_config()

def option_1():
    # make clicking on the main window impossible
    root.attributes('-disabled', True)
    window = tk.Toplevel(root)
    window.title('Options')
    window.geometry('400x100')
    window.configure(bg='black')
    window.geometry("+{}+{}".format(positionRight, positionDown))
    # label 'Current translator:'
    # if english is the current language, the label will be 'Current tranlation model:'
    # if vietnamese is the current language, the label will be 'Mô hình dịch hiện tại:'
    if language == 'English':
        current_translator_label = tk.Label(window, text='Current translation model:', bg='black', fg='white')
    elif language == 'Vietnamese':
        current_translator_label = tk.Label(window, text='Mô hình dịch hiện tại:', bg='black', fg='white')
    elif language == 'Chinese':
        current_translator_label = tk.Label(window, text='当前翻译模型:', bg='black', fg='white')
    current_translator_label.grid(row=0, column=0, padx=10, pady=10)
    # combobox
    current_translator = tk.StringVar()
    # set the current translator to the last one in the list
    modl_name, modl_dir = get_model()
    current_translator.set(modl_name[-1])
    translator_combobox = ttk.Combobox(window, textvariable=current_translator, values=modl_name, state='readonly')
    translator_combobox.grid(row=0, column=1, padx=10, pady=10)
    
    # add button 'Add'
    def add():
        # open the text file and get the list of model paths
        modl_name, modl_dir = get_model()
        # opne a file dialog and choose model folder
        if language == 'English':
            file = filedialog.askdirectory(initialdir=parent_dir, title='Select model folder')
        elif language == 'Vietnamese':
            file = filedialog.askdirectory(initialdir=parent_dir, title='Chọn mô hình thư mục')
        elif language == 'Chinese':
            file = filedialog.askdirectory(initialdir=parent_dir, title='选择模型文件夹')
        # get the name of the model
        name = file.split('/')[-1].split('.')[0]
        # add the name to the combobox
        translator_combobox['values'] = modl_name + [name]
        # set the current translator to the new one
        current_translator.set(name)
        # save the name and its path to the text file (append to new line without using '\n')
        with open(Path(__file__).with_name('models.txt'), 'a') as f:
            f.write(name + ', ' + file + '\n')

        # update the list of model paths
        modl_name, modl_dir = get_model()
        print(modl_dir)
        # configure the combobox
        translator_combobox.configure(values=modl_name + [name])

    if language == 'English':
        add_button = tk.Button(window, text='Add', bg='black', fg='white', command=add)
    elif language == 'Vietnamese':
        add_button = tk.Button(window, text='Thêm', bg='black', fg='white', command=add)
    elif language == 'Chinese':
        add_button = tk.Button(window, text='加', bg='black', fg='white', command=add)
    add_button.grid(row=0, column=2, padx=10, pady=10)

    # button 'Change'
    def change(): 
        modl_name, modl_dir = get_model()
        fkin_name_index = modl_name.index(current_translator.get())
        # remove the model name and its path from the text file
        with open(Path(__file__).with_name('models.txt'), 'r') as f:
            lines = f.readlines()
        with open(Path(__file__).with_name('models.txt'), 'w') as f:
            for line in lines:
                if not line.startswith(modl_name[fkin_name_index]):
                    f.write(line)
        # add the model name and its path to the text file
        with open(Path(__file__).with_name('models.txt'), 'a') as f:
            f.write(modl_name[fkin_name_index] + ', ' + modl_dir[fkin_name_index] + '\n')
        # model switch
        model_switch()
        # configure the combobox
        translator_combobox.configure(values=modl_name)
        window.destroy()
        print(model)
        return model, tokenizer

    if language == 'English':
        change_button = tk.Button(window, text='Change', bg='black', fg='white', command=change)
    elif language == 'Vietnamese':
        change_button = tk.Button(window, text='Thay đổi', bg='black', fg='white', command=change)
    elif language == 'Chinese':
        change_button = tk.Button(window, text='更改', bg='black', fg='white', command=change)
    change_button.grid(row=1, columnspan=2, padx=10, pady=10)

    root.attributes('-disabled', False)
    window.mainloop()

def option_2():
    # make clicking on the main window impossible
    root.attributes('-disabled', True)
    window = tk.Toplevel(root)
    window.title('Options')
    window.geometry('400x100')
    window.configure(bg='black')
    window.geometry("+{}+{}".format(positionRight, positionDown))
    # label 'Import method:'
    if language == 'English':
        import_method_label = tk.Label(window, text='Import method:', bg='black', fg='white')
    elif language == 'Vietnamese':
        import_method_label = tk.Label(window, text='Cách nhập tệp:', bg='black', fg='white')
    elif language == 'Chinese':
        import_method_label = tk.Label(window, text='导入方法:', bg='black', fg='white')
    import_method_label.grid(row=0, column=0, padx=10, pady=10)
    # combobox
    import_method = tk.StringVar()
    if language == 'English':
        import_method.set('Choose...')
    elif language == 'Vietnamese':
        import_method.set('Chọn...')
    elif language == 'Chinese':
        import_method.set('选择...')
    import_method_combobox = ttk.Combobox(window, textvariable=import_method, values=['csv', 'xlsx', 'txt'], state='readonly')
    import_method_combobox.grid(row=0, column=1, padx=10, pady=10)
    
    # csv method is the default method
    def if_csv():
        # Choose CSV file label
        if language == 'English':
            csv_label = tk.Label(tab2, text='Choose csv file:', bg='black', fg='white')
        elif language == 'Vietnamese':
            csv_label = tk.Label(tab2, text='Chọn tệp csv:', bg='black', fg='white')
        elif language == 'Chinese':
            csv_label = tk.Label(tab2, text='选择csv文件:', bg='black', fg='white')
        csv_label.grid(row=0, column=0, padx=15, pady=20)

        # a entry for the CSV file path
        csv_entry = tk.Entry(tab2, width=45, bg='white', font=('Segoe UI', 10, 'normal'))
        csv_entry.grid(row=0, column=1, padx=10, pady=20, sticky='ew')

        # choose CSV file button
        if language == 'English':
            csv_button = tk.Button(tab2, text='Choose', bg='black', fg='white', font=('Segoe UI', 10, 'bold'), command=choose_csv)
        elif language == 'Vietnamese':
            csv_button = tk.Button(tab2, text='Chọn', bg='black', fg='white', font=('Segoe UI', 10, 'bold'), command=choose_csv)
        elif language == 'Chinese':
            csv_button = tk.Button(tab2, text='选择', bg='black', fg='white', font=('Segoe UI', 10, 'bold'), command=choose_csv)
        csv_button.grid(row=0, column=2, padx=10, pady=20)

        # translate button (by configuring the command of the translate button)
        translate_button_csv.configure(command=translate_csv)

    # xlsx method
    def if_xlsx():
        global xlsx_entry, translate_button_xlsx
        # Choose xlsx file label
        if language == 'English':      
            xlsx_label = tk.Label(tab2, text='Choose xlsx file:', bg='black', fg='white')
        elif language == 'Vietnamese':
            xlsx_label = tk.Label(tab2, text='Chọn tệp xlsx:', bg='black', fg='white')
        elif language == 'Chinese':
            xlsx_label = tk.Label(tab2, text='选择xlsx文件:', bg='black', fg='white')
        xlsx_label.grid(row=0, column=0, padx=15, pady=20)

        # a entry for the xlsx file path
        xlsx_entry = tk.Entry(tab2, width=45, bg='white', font=('Segoe UI', 10, 'normal'))
        xlsx_entry.grid(row=0, column=1, padx=10, pady=20, sticky='ew')

        # choose xlsx file button
        if language == 'English':
            xlsx_button = tk.Button(tab2, text='Choose', bg='black', fg='white', font=('Segoe UI', 10, 'bold'), command=choose_xlsx)
        elif language == 'Vietnamese':
            xlsx_button = tk.Button(tab2, text='Chọn', bg='black', fg='white', font=('Segoe UI', 10, 'bold'), command=choose_xlsx)
        elif language == 'Chinese':
            xlsx_button = tk.Button(tab2, text='选择', bg='black', fg='white', font=('Segoe UI', 10, 'bold'), command=choose_xlsx)
        xlsx_button.grid(row=0, column=2, padx=10, pady=20)

        # translate button (by configuring the command of the translate button)
        translate_button_xlsx = translate_button_csv
        translate_button_xlsx.configure(command=translate_xlsx)

    # txt method
    def if_txt():
        global txt_entry, translate_button_txt
        # Choose txt file label
        if language == 'English':
            txt_label = tk.Label(tab2, text='Choose txt file:', bg='black', fg='white')
        elif language == 'Vietnamese':
            txt_label = tk.Label(tab2, text='Chọn tệp txt:', bg='black', fg='white')
        elif language == 'Chinese':
            txt_label = tk.Label(tab2, text='选择txt文件:', bg='black', fg='white')
        txt_label.grid(row=0, column=0, padx=15, pady=20)

        # a entry for the txt file path
        txt_entry = tk.Entry(tab2, width=45, bg='white', font=('Segoe UI', 10, 'normal'))
        txt_entry.grid(row=0, column=1, padx=10, pady=20, sticky='ew')

        # choose txt file button
        if language == 'English':
            txt_button = tk.Button(tab2, text='Choose', bg='black', fg='white', font=('Segoe UI', 10, 'bold'), command=choose_txt)
        elif language == 'Vietnamese':
            txt_button = tk.Button(tab2, text='Chọn', bg='black', fg='white', font=('Segoe UI', 10, 'bold'), command=choose_txt)
        elif language == 'Chinese':
            txt_button = tk.Button(tab2, text='选择', bg='black', fg='white', font=('Segoe UI', 10, 'bold'), command=choose_txt)
        txt_button.grid(row=0, column=2, padx=10, pady=20)

        # translate button (by configuring the command of the translate button)
        translate_button_txt = translate_button_csv
        translate_button_txt.configure(command=translate_txt)

    # button 'Apply'
    def apply():
        if import_method.get() == 'csv':
            if_csv()
        elif import_method.get() == 'xlsx':
            if_xlsx()
        elif import_method.get() == 'txt':
            if_txt()
        window.destroy()

    if language == 'English':
        apply_button = tk.Button(window, text='Apply', bg='black', fg='white', command=apply)
    elif language == 'Vietnamese':
        apply_button = tk.Button(window, text='Áp dụng', bg='black', fg='white', command=apply)
    elif language == 'Chinese':
        apply_button = tk.Button(window, text='应用', bg='black', fg='white', command=apply)
    apply_button.grid(row=1, columnspan=2, padx=10, pady=10)

    root.attributes('-disabled', False)
    window.mainloop()
        
# Choose CSV file button
def choose_csv():
    if language == 'English':
        csv_file = filedialog.askopenfilename(initialdir=parent_dir, title='Select CSV file', filetypes=(('CSV files', '*.csv'),))
    elif language == 'Vietnamese':
        csv_file = filedialog.askopenfilename(initialdir=parent_dir, title='Chọn tệp CSV', filetypes=(('CSV files', '*.csv'),))
    elif language == 'Chinese':
        csv_file = filedialog.askopenfilename(initialdir=parent_dir, title='选择CSV文件', filetypes=(('CSV files', '*.csv'),))
    csv_entry.delete(0, 'end')
    csv_entry.insert(0, csv_file)

# Choose xlsx file button
def choose_xlsx():
    if language == 'English':
        xlsx_file = filedialog.askopenfilename(initialdir=parent_dir, title='Select xlsx file', filetypes=(('xlsx files', '*.xlsx'),))
    elif language == 'Vietnamese':
        xlsx_file = filedialog.askopenfilename(initialdir=parent_dir, title='Chọn tệp xlsx', filetypes=(('xlsx files', '*.xlsx'),))
    elif language == 'Chinese':
        xlsx_file = filedialog.askopenfilename(initialdir=parent_dir, title='选择xlsx文件', filetypes=(('xlsx files', '*.xlsx'),))
    xlsx_entry.delete(0, 'end')
    xlsx_entry.insert(0, xlsx_file)

# Choose txt file button
def choose_txt():
    if language == 'English':
        txt_file = filedialog.askopenfilename(initialdir=parent_dir, title='Select txt file', filetypes=(('txt files', '*.txt'),))
    elif language == 'Vietnamese':
        txt_file = filedialog.askopenfilename(initialdir=parent_dir, title='Chọn tệp txt', filetypes=(('txt files', '*.txt'),))
    elif language == 'Chinese':
        txt_file = filedialog.askopenfilename(initialdir=parent_dir, title='选择txt文件', filetypes=(('txt files', '*.txt'),))
    txt_entry.delete(0, 'end')
    txt_entry.insert(0, txt_file)

def translate_and_save(csv):
    global progress, progress_label
    try:
        # Read the CSV file
        df = pd.read_csv(csv, encoding='utf-8-sig', names=['Chingchong'], header=None)
        df['Vietnamese'] = ''

        for i, row in df.iterrows():
            # Translate the text
            if 'gpt' in model_dir:
                out = translate_gpt(row['Chingchong'], model)
            else:
                out = translate_hugging(row['Chingchong'])
            # Save the translated text to the column 'Vietnamese'
            df.at[i, 'Vietnamese'] = out
            # Update the progress bar
            progress['value'] = (i + 1) / len(df) * 100
            progress_label['text'] = f'{i + 1}/{len(df)}'
            progress.update()
    
    except Exception as e:
        if language == 'English':
            messagebox.showerror('Error', f'Something went wrong: {str(e)}')
        elif language == 'Vietnamese':
            messagebox.showerror('Lỗi', f'Có gì đó không đúng: {str(e)}')
        elif language == 'Chinese':
            messagebox.showerror('错误', f'出了点问题: {str(e)}')
        translate_button_csv.configure(state='normal')
    
    # add 'Translated' in csv path
    csv = csv.split('.csv')[0] + '_translated.csv'
    # save the translated column to the CSV file
    df.to_csv(csv, encoding='utf-8-sig', index=False, header=False)
    # show a message box when the translation is done
    if language == 'English':
        messagebox.showinfo('Done', 'Translation is complete.')
    elif language == 'Vietnamese':
        messagebox.showinfo('Hoàn thành', 'Dịch đã hoàn tất.')
    elif language == 'Chinese':
        messagebox.showinfo('完成', '翻译完成。')
    # open the translated CSV file
    return csv

# Translate CSV file
def translate_csv():
    global progress, progress_label
    translate_button_csv.configure(state='disabled')
    # get the path of the CSV file
    csv_file = csv_entry.get()
    # create a progress bar that shows the progress of the translation process based on the number of lines in the CSV file
    progress = ttk.Progressbar(tab2, orient=HORIZONTAL, length=250, mode='determinate')
    progress.grid(row=2, column=1, padx=10, pady=20)
    # and the label 'Progressing: i/n lines'
    if language == 'English':
        progress_label = tk.Label(tab2, text='Progressing...', bg='black', fg='white')
    elif language == 'Vietnamese':
        progress_label = tk.Label(tab2, text='Đang tiến hành...', bg='black', fg='white')
    elif language == 'Chinese':
        progress_label = tk.Label(tab2, text='进展中...', bg='black', fg='white')
    progress_label.grid(row=3, column=1, padx=10, pady=0)
    time.sleep(1)
    # start the translation process
    csv = translate_and_save(csv_file)
    # enable the translate button
    translate_button_csv.configure(state='normal')
    progress.grid_forget()
    progress_label.grid_forget()
    # open the translated CSV file
    os.startfile(csv)

def translate_xlsx():
    global xlsx_entry, translate_button_xlsx, progress, progress_label
    translate_button_xlsx.configure(state='disabled')
    # get the path of the xlsx file
    xlsx_file = xlsx_entry.get()
    # get the file name without the extension
    file_name = xlsx_file.split('/')[-1].split('.')[0]
    # get the path of the xlsx file without the file name
    path = '/'.join(xlsx_file.split('/')[:-1]) + '/'
    # convert the xlsx file to a CSV file
    df = pd.read_excel(xlsx_file, names=['Chingchong'], header=None)
    temp = path + 'temp.csv'
    df.to_csv(temp, encoding='utf-8-sig', index=False, header=False)
    # create a progress bar that shows the progress of the translation process based on the number of lines in the CSV file
    progress = ttk.Progressbar(tab2, orient=HORIZONTAL, length=250, mode='determinate')
    progress.grid(row=2, column=1, padx=10, pady=20)
    # and the label 'Progressing: i/n lines'
    if language == 'English':
        progress_label = tk.Label(tab2, text='Progressing...', bg='black', fg='white')
    elif language == 'Vietnamese':
        progress_label = tk.Label(tab2, text='Đang tiến hành...', bg='black', fg='white')
    elif language == 'Chinese':
        progress_label = tk.Label(tab2, text='进展中...', bg='black', fg='white')
    progress_label.grid(row=3, column=1, padx=10, pady=0)
    time.sleep(1)
    # translate the CSV file
    temp_tl = translate_and_save(temp)
    # remove the existing file if there is one
    if os.path.exists(path + file_name + '_translated.csv'):
        os.remove(path + file_name + '_translated.csv')
    # rename temp back to the original file name
    csv = Path(temp_tl).rename(path + file_name + '_translated.csv')
    Path(temp).unlink()
    # enable the translate button
    translate_button_xlsx.configure(state='normal')
    progress.grid_forget()
    progress_label.grid_forget()
    # open the translated CSV file
    os.startfile(csv)

def translate_txt():
    global txt_entry, translate_button_txt, progress, progress_label
    translate_button_txt.configure(state='disabled')
    # get the path of the txt file
    txt_file = txt_entry.get()
    # get the file name without the extension
    file_name = txt_file.split('/')[-1].split('.')[0]
    # get the path of the txt file without the file name
    path = '/'.join(txt_file.split('/')[:-1]) + '/'
    # convert the txt file to a csv file
    with open(txt_file, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        # remove new line characters
        lines = [i.strip() for i in lines]
        # convert the list of lines to a dataframe
        df = pd.DataFrame(lines, columns=['Chingchong'], dtype=str, index=None)
        # save the dataframe to a csv file
        temp = path + 'temp.csv'
        df.to_csv(temp, encoding='utf-8-sig', index=False, header=False)
    # create a progress bar that shows the progress of the translation process based on the number of lines in the CSV file
    progress = ttk.Progressbar(tab2, orient=HORIZONTAL, length=250, mode='determinate')
    progress.grid(row=2, column=1, padx=10, pady=20)
    # and the label 'Progressing: i/n lines'
    if language == 'English':
        progress_label = tk.Label(tab2, text='Progressing...', bg='black', fg='white')
    elif language == 'Vietnamese':
        progress_label = tk.Label(tab2, text='Đang tiến hành...', bg='black', fg='white')
    elif language == 'Chinese':
        progress_label = tk.Label(tab2, text='进展中...', bg='black', fg='white')
    progress_label.grid(row=3, column=1, padx=10, pady=0)
    time.sleep(1)
    # translate the CSV file
    temp_tl = translate_and_save(temp)
    # remove the existing file if there is one
    if os.path.exists(path + file_name + '_translated.csv'):
        os.remove(path + file_name + '_translated.csv')
    # rename temp back to the original file name
    csv = Path(temp_tl).rename(path + file_name + '_translated.csv')
    # delete the temporary CSV file
    Path(temp).unlink()
    # enable the translate button
    translate_button_txt.configure(state='normal')
    progress.grid_forget()
    progress_label.grid_forget()
    # open the translated CSV file
    os.startfile(csv)

def eng_config():
    global language
    # Configure everything to English
    root.title('Chingchong to Vietnamese')
    input_label.configure(text='Input:')
    translate_button.configure(text='Translate')
    output_label.configure(text='Output:')
    csv_label.configure(text='Choose csv file:')
    csv_button.configure(text='Choose')
    translate_button_csv.configure(text='Translate')
    csv_entry.configure(width=45)
    csv_entry.grid(row=0, column=1, padx=10, pady=20, sticky='ew')
    translate_button.configure(width=20)
    translate_button.grid(columnspan=3, padx=5, pady=5)
    output_box.configure(height=6, width=60)
    output_box.pack(expand=True, fill='both')
    tab_control.tab(0, text='Regular translation')
    tab_control.tab(1, text='File translation')
    optionsmenu.entryconfigure(0, label='Change translators...')
    optionsmenu.entryconfigure(1, label='Change file import methods...')
    optionsmenu.entryconfigure(2, label='Language')
    filemenu.entryconfigure(0, label='Exit')
    language = 'English'
    # remove '\nEnglish' from lang.txt and append it to the end of the file
    with open(Path(__file__).with_name('lang.txt'), 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines if line.strip() != language]
    lines.append(language)
    with open(Path(__file__).with_name('lang.txt'), 'w') as file:
        file.write('\n'.join(lines))
    return language

def viet_config():
    global language
    # Configure everything to Vietnamese
    root.title('Chingchong to Vietnamese')
    input_label.configure(text='Nhập:')
    translate_button.configure(text='Dịch')
    output_label.configure(text='Kết quả:')
    csv_label.configure(text='Chọn tệp csv:')
    csv_button.configure(text='Chọn')
    translate_button_csv.configure(text='Dịch')
    csv_entry.configure(width=45)
    csv_entry.grid(row=0, column=1, padx=10, pady=20, sticky='ew')
    translate_button.configure(width=20)
    translate_button.grid(columnspan=3, padx=5, pady=5)
    output_box.configure(height=6, width=60)
    output_box.pack(expand=True, fill='both')
    tab_control.tab(0, text='Dịch thông thường')
    tab_control.tab(1, text='Dịch file')
    optionsmenu.entryconfigure(0, label='Thay đổi mô hình dịch...')
    optionsmenu.entryconfigure(1, label='Thay đổi loại tệp nhập vào...')
    optionsmenu.entryconfigure(2, label='Ngôn ngữ')
    filemenu.entryconfigure(0, label='Thoát')
    language = 'Vietnamese'
    # remove '\nVietnamese' from lang.txt and write it to the end of the file
    with open(Path(__file__).with_name('lang.txt'), 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines if line.strip() != language]
    lines.append(language)
    with open(Path(__file__).with_name('lang.txt'), 'w') as file:
        file.write('\n'.join(lines))
    return language

def china_config():
    global language
    # Configure everything to Chinese
    root.title('Chingchong to Vietnamese')
    input_label.configure(text='输入：')
    translate_button.configure(text='翻译')
    output_label.configure(text='输出：')
    csv_label.configure(text='选择csv文件：')
    csv_button.configure(text='选择')
    translate_button_csv.configure(text='翻译')
    csv_entry.configure(width=45)
    csv_entry.grid(row=0, column=1, padx=10, pady=20, sticky='ew')
    translate_button.configure(width=20)
    translate_button.grid(columnspan=3, padx=5, pady=5)
    output_box.configure(height=6, width=60)
    output_box.pack(expand=True, fill='both')
    tab_control.tab(0, text='常规翻译')
    tab_control.tab(1, text='文件翻译')
    optionsmenu.entryconfigure(0, label='更改翻译模型...')
    optionsmenu.entryconfigure(1, label='更改文件导入方法...')
    optionsmenu.entryconfigure(2, label='语言')
    filemenu.entryconfigure(0, label='退出')
    language = 'Chinese'
    # remove 'Chinese' from lang.txt and append it to the end of the file
    with open(Path(__file__).with_name('lang.txt'), 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines if line.strip() != language]
    lines.append(language)
    with open(Path(__file__).with_name('lang.txt'), 'w') as file:
        file.write('\n'.join(lines))
    return language

# root window
root = tk.Tk()
root.title('Chingchong to Vietnamese')
root.geometry('600x380')
root.configure(bg='black')

# position the window center in the center of the screen
positionRight = int(root.winfo_screenwidth() / 2 - 600 / 2)
positionDown = int(root.winfo_screenheight() / 2 - 380 / 2)
root.geometry("+{}+{}".format(positionRight, positionDown))

# Menu bar
menu = Menu(root)
root.config(menu=menu)

# file
filemenu = Menu(menu, tearoff=0)
filemenu.add_command(label='Exit', command=root.quit)
menu.add_cascade(label='File', menu=filemenu)

# options
optionsmenu = Menu(menu, tearoff=0)
optionsmenu.add_command(label='Change translators...', command=option_1)
optionsmenu.add_command(label='Change file import methods...', command=option_2)
# options language with submenus of it
optionsmenu_lang = Menu(optionsmenu, tearoff=0)
optionsmenu_lang.add_command(label='English', command=eng_config)
optionsmenu_lang.add_command(label='Tiếng Việt', command=viet_config)
optionsmenu_lang.add_command(label='中文', command=china_config)
optionsmenu.add_cascade(label='Language', menu=optionsmenu_lang)

menu.add_cascade(label='Options', menu=optionsmenu)

# first tab is regular translation and second tab is CSV translation
tab_control = ttk.Notebook(root)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab_control.add(tab1, text='Regular translation')
tab_control.add(tab2, text='File translation')
tab_control.pack(expand=1, fill='both')
sth = ttk.Style()
sth.configure('TNotebook', background='black', foreground='white')
# style for tab
sth.configure('TFrame', background='black')

# tab1

# input label
input_label = tk.Label(tab1, text='Input:', bg='black', fg='white', font=('Segoe UI', 10, 'normal'))
input_label.grid(row=0, column=0, padx=30, pady=20)

# a frame for the input box
input_frame = tk.Frame(tab1, bg='black')
input_frame.grid(row=0, column=1, padx=10, pady=20, sticky='ew')

# input box
input_box = tk.Text(input_frame, height=6, width=60, bg='white', font=('Segoe UI', 10, 'normal'))
input_box.pack(expand=True, fill='both')

# translate button (in the center)
translate_button = tk.Button(tab1, text='Translate', bg='black', fg='white', font=('Segoe UI', 10, 'bold'), command=lambda: translate(input_box.get('1.0', 'end')), width=20)
translate_button.grid(columnspan=3, padx=5, pady=5)

# output label
output_label = tk.Label(tab1, text='Output:', bg='black', fg='white', font=('Segoe UI', 10, 'normal'))
output_label.grid(row=2, column=0, padx=30, pady=20)

# a frame for the output box
output_frame = tk.Frame(tab1, bg='black')
output_frame.grid(row=2, column=1, padx=10, pady=20, sticky='ew')

# output box
output_box = tk.Text(output_frame, height=6, width=60, bg='white', font=('Segoe UI', 10, 'normal'), state='disabled')
output_box.pack(expand=True, fill='both')

# tab2

# Choose CSV file label
csv_label = tk.Label(tab2, text='Choose csv file:', bg='black', fg='white', font=('Segoe UI', 10, 'normal'))
csv_label.grid(row=0, column=0, padx=15, pady=20)

# a entry for the CSV file path
csv_entry = tk.Entry(tab2, width=45, bg='white', font=('Segoe UI', 10, 'normal'))
csv_entry.grid(row=0, column=1, padx=10, pady=20, sticky='ew')

# choose CSV file button
csv_button = tk.Button(tab2, text='Choose', bg='black', fg='white', font=('Segoe UI', 10, 'bold'), command=choose_csv)
csv_button.grid(row=0, column=2, padx=10, pady=20)

# translate button
translate_button_csv = tk.Button(tab2, text='Translate', bg='black', fg='white', font=('Segoe UI', 10, 'bold'), command=translate_csv, width=20)
translate_button_csv.grid(column=1, padx=5, pady=5)

# language initialization
language_switch()

root.mainloop()