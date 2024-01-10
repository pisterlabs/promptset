import tkinter as tk
from tkinter import ttk  # 注意這個導入
from tkinter import filedialog
import openai
from tqdm import tqdm  # 新增的進度條套件

prompt = """翻譯成繁體中文"""

def translate(text, temp, api_key, Model):
    openai.api_key = api_key
    translation = ""
    if Model == "gpt-4":
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                "role": "system",
                "content": "完整翻譯成繁體中文"
                },
                {
                "role": "user",
                "content": text
                }
            ],
            temperature=temp,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        translation = response.choices[0]["message"]["content"]
    if Model == "text-davinci-003":
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"完整翻譯成繁體中文.: {text}",
            temperature=0.1,
            max_tokens=200,
            stop="\t"
        )
        translation = response.choices[0].text.strip()
    return translation

def translate_selected_text(event=None):
    selected_index = original_listbox.curselection()  # 獲取使用者選取的索引
    if selected_index:
        selected_text = original_listbox.get(selected_index)  # 獲取使用者選取的文字
        temperature = float(temperature_var.get())  # 獲取使用者輸入的temperature值
        api_key = api_key_var.get()  # 獲取使用者輸入的API金鑰
        model = selected_option.get()
        if not api_key:
            tk.messagebox.showwarning("警告", "請輸入API金鑰！")
            return
        translation = translate(selected_text, temperature, api_key, model)  # 翻譯成中文
        translated_listbox.delete(selected_index)  # 刪除對應行的內容
        translated_listbox.insert(selected_index, translation)  # 將翻譯結果插入對應行

def translate_text_with_progress(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        total_lines = sum(1 for line in file)
        file.seek(0)
        progress_bar["maximum"] = total_lines
        for idx, line in enumerate(file):
            original_listbox.insert(tk.END, line.strip())
            temperature = float(temperature_var.get())
            api_key = api_key_var.get()
            model = selected_option.get()
            if not api_key:
                tk.messagebox.showwarning("警告", "請輸入API金鑰！")
                return
            translation = translate(line, temperature, api_key, model)
            translated_listbox.insert(tk.END, translation)
            progress_bar["value"] = idx + 1
            root.update()  # 更新GUI，以使進度條反應更流暢

def translate_text():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        original_listbox.delete(0, tk.END)
        translated_listbox.delete(0, tk.END)
        progress_bar["value"] = 0  # 重置進度條
        translate_text_with_progress(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            original_listbox.delete(0, tk.END)
            translated_listbox.delete(0, tk.END)  # 清空翻譯結果Listbox
            for line in file:
                original_listbox.insert(tk.END, line.strip())
                temperature = float(temperature_var.get())  # 獲取使用者輸入的temperature值
                api_key = api_key_var.get()  # 獲取使用者輸入的API金鑰
                model = selected_option.get()
                if not api_key:
                    tk.messagebox.showwarning("警告", "請輸入API金鑰！")
                    return
                translation = translate(line, temperature, api_key, model)  # 翻譯成中文
                translated_listbox.insert(tk.END, translation)

def export_translations():
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, 'w', encoding='utf-8') as file:
            for i in range(original_listbox.size()):
                original_text = original_listbox.get(i)
                translated_text = translated_listbox.get(i)
                file.write(f"{original_text}\t{translated_text}\n")

# 建立主視窗
root = tk.Tk()
root.title("Text Translation Tool")

# Listbox顯示原文
original_listbox = tk.Listbox(root, selectbackground="lightyellow", selectmode=tk.SINGLE, width=40, height=20)
original_listbox.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

# 捲軸條
scrollbar = tk.Scrollbar(root, command=original_listbox.yview)
scrollbar.pack(side=tk.LEFT, fill=tk.Y)
original_listbox.config(yscrollcommand=scrollbar.set)

# Listbox顯示翻譯結果
translated_listbox = tk.Listbox(root, selectbackground="lightblue", selectmode=tk.SINGLE, width=40, height=20)
translated_listbox.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

# 捲軸條
scrollbar = tk.Scrollbar(root, command=translated_listbox.yview)
scrollbar.pack(side=tk.LEFT, fill=tk.Y)
translated_listbox.config(yscrollcommand=scrollbar.set)


# 創建進度條
progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
progress_bar.pack(pady=10)

# 按鈕
translate_button = tk.Button(root, text="翻譯選取的文字", command=translate_selected_text)
translate_button.pack(pady=5)

load_file_button = tk.Button(root, text="載入檔案", command=translate_text)
load_file_button.pack(pady=5)

export_button = tk.Button(root, text="導出翻譯結果", command=export_translations)
export_button.pack(pady=5)

# 翻譯結果標籤
translated_text = tk.StringVar()
translated_label = tk.Label(root, textvariable=translated_text)
translated_label.pack(pady=10)


# 創建一個StringVar對象來保存選擇的選項
selected_option = tk.StringVar(root)

# 設置默認選擇的選項
selected_option.set("text-davinci-003")

# 創建一個OptionMenu對象，並將選擇的選項保存到selected_option中
dropdown = tk.OptionMenu(root, selected_option, "text-davinci-003", "gpt-4")
dropdown.pack()


# Prompt輸入
prompt_label = tk.Label(root, text="Prompt:")
prompt_label.pack()

prompt_var = tk.StringVar()
prompt_entry = tk.Entry(root, width=40, textvariable=prompt_var)
prompt_entry.pack()

# Temperature輸入
temperature_label = tk.Label(root, text="Temperature:")
temperature_label.pack()

temperature_var = tk.DoubleVar()
temperature_entry = tk.Entry(root, width=10, textvariable=temperature_var)
temperature_entry.pack()

# API金鑰輸入
api_key_label = tk.Label(root, text="API金鑰:")
api_key_label.pack()


api_key_var = tk.StringVar()
api_key_entry = tk.Entry(root, width=40, textvariable=api_key_var)
api_key_entry.pack()

root.mainloop()