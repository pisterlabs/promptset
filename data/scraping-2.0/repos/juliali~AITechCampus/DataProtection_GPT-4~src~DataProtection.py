import tkinter as tk
import csv
import spacy
import secrets
import openai
import re


class UIApp:
    def __init__(self):
        with open('key.info', 'r') as file:
            self.api_key = file.readline().strip()

        self.colorconfig = {}
        with open('entitycolor.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.colorconfig[row[0]] = row[1]

    def clear_text(self, text_widget):
        text_widget.delete(1.0, tk.END)

    def clear_all(self, input_text, replace_text, rewrite_text, result_text):
        input_text.delete(1.0, tk.END)
        replace_text.delete(1.0, tk.END)
        rewrite_text.delete(1.0, tk.END)
        result_text.delete(1.0, tk.END)

    def create_UI(self):
        root = tk.Tk()
        root.title("Data Protection")

        input_text = tk.Text(root, width=80, height=10, bg=self.colorconfig.get('input_bg', 'white'))
        input_text.grid(row=0, column=0, columnspan=2)

        process_btn = tk.Button(root, text="Process", command=lambda: self.process_input(input_text, replace_text))
        process_btn.grid(row=1, column=0, sticky='e')

        clear_input_btn = tk.Button(root, text="Clear", command=lambda: self.clear_text(input_text))
        clear_input_btn.grid(row=1, column=1, sticky='w', padx=(8, 0))  # 添加 padx 参数

        replace_text = tk.Text(root, width=80, height=10, bg=self.colorconfig.get('replace_bg', 'white'))
        replace_text.grid(row=2, column=0, columnspan=2)

        transform_btn = tk.Button(root, text="Transform",
                                  command=lambda: self.transform_replaced(replace_text, rewrite_text))
        transform_btn.grid(row=3, column=0, sticky='e')

        clear_replace_btn = tk.Button(root, text="Clear", command=lambda: self.clear_text(replace_text))
        clear_replace_btn.grid(row=3, column=1, sticky='w', padx=(8, 0))  # 添加 padx 参数

        rewrite_text = tk.Text(root, width=80, height=10, bg=self.colorconfig.get('rewrite_bg', 'white'))
        rewrite_text.grid(row=4, column=0, columnspan=2)

        unprocess_btn = tk.Button(root, text="Unprocess",
                                  command=lambda: self.unprocess_transformed(rewrite_text, result_text))
        unprocess_btn.grid(row=5, column=0, sticky='e')

        clear_rewrite_btn = tk.Button(root, text="Clear", command=lambda: self.clear_text(rewrite_text))
        clear_rewrite_btn.grid(row=5, column=1, sticky='w', padx=(8, 0))  # 添加 padx 参数

        result_text = tk.Text(root, width=80, height=10, bg=self.colorconfig.get('result_bg', 'white'))
        result_text.grid(row=6, column=0, columnspan=2)

        clear_all_btn = tk.Button(root, text="ClearAll", command=lambda: self.clear_all(input_text, replace_text, rewrite_text, result_text))
        clear_all_btn.grid(row=7, column=0, columnspan=2)

        root.mainloop()


    def process_input(self, input_text_widget, replace_text_widget):
        nlp = spacy.load("en_core_web_sm")
        input_text = input_text_widget.get(1.0, tk.END).strip()
        doc = nlp(input_text)

        input_text_widget.delete(1.0, tk.END)
        replace_text_widget.delete(1.0, tk.END)

        prev_ent_end_char = 0
        replacement_list = []
        for ent in doc.ents:
            random_str = secrets.token_hex(4)

            input_text_widget.insert(tk.END, doc.text[prev_ent_end_char:ent.start_char])
            entity_color = self.colorconfig.get(ent.label_, 'black')
            input_text_widget.insert(tk.END, ent.text, 'color_{}'.format(entity_color))

            replace_text_widget.insert(tk.END, doc.text[prev_ent_end_char:ent.start_char])
            replace_text_widget.insert(tk.END, random_str, 'color_{}'.format(entity_color))

            prev_ent_end_char = ent.end_char
            replacement_list.append((ent.text, random_str, entity_color))

        input_text_widget.insert(tk.END, doc.text[prev_ent_end_char:])
        replace_text_widget.insert(tk.END, doc.text[prev_ent_end_char:])

        for ent_type, color in self.colorconfig.items():
            input_text_widget.tag_configure('color_{}'.format(color), foreground=color)
            replace_text_widget.tag_configure('color_{}'.format(color), foreground=color)

        with open('replacements.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for original, replacement, color in replacement_list:
                writer.writerow([original, replacement, color])

    def transform_replaced(self, replace_text_widget, rewritten_text_widget):
        # 设置API密钥并配置OpenAI库
        openai.api_key = self.api_key
        model_engine = "text-davinci-002"

        # 从替换文本框中获取文本
        replace_text = replace_text_widget.get(1.0, tk.END).strip()

        # 构建GPT-3 API调用的prompt
        prompt = f"Instruction: Rewrite to a lecture notes; Input: {replace_text}"

        # 调用GPT-3 API
        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.5,
        )

        # 获取API响应并插入到改写文本框中
        rewritten_text = response.choices[0].text.strip()
        rewritten_text_widget.delete(1.0, tk.END)
        rewritten_text_widget.insert(tk.END, rewritten_text)

        # 读取replacements.csv文件并为改写文本框中的对应字符串添加颜色
        with open('replacements.csv', 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for original, replacement, color in reader:
                pattern = re.compile(re.escape(replacement))
                for match in pattern.finditer(rewritten_text):
                    rewritten_text_widget.tag_configure('color_{}'.format(color), foreground=color)

                    start = rewritten_text_widget.search(match.group(0), 1.0, stopindex=tk.END)
                    if start:
                        end = f"{start.split('.')[0]}.{int(start.split('.')[1]) + len(match.group(0))}"
                        rewritten_text_widget.tag_add('color_{}'.format(color), start, end)


    def unprocess_transformed(self, rewritten_text_widget, result_text_widget):
        # 获取改写文本框中的文本
        rewritten_text = rewritten_text_widget.get(1.0, tk.END).strip()

        # 读取replacements.csv文件并为改写文本框中的对应字符串添加颜色
        replacements = []
        with open('replacements.csv', 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for original, replacement, color in reader:
                replacements.append((original, replacement, color))

        # 按照替换顺序反向替换文本
        for original, replacement, color in reversed(replacements):
            rewritten_text = rewritten_text.replace(replacement, original)

        # 将结果插入结果文本框中并为对应字符串添加颜色
        result_text_widget.delete(1.0, tk.END)
        result_text_widget.insert(tk.END, rewritten_text)

        for original, replacement, color in replacements:
            pattern = re.compile(re.escape(original))
            for match in pattern.finditer(rewritten_text):
                result_text_widget.tag_configure('color_{}'.format(color), foreground=color)

                start = result_text_widget.search(match.group(0), 1.0, stopindex=tk.END)
                if start:
                    end = f"{start.split('.')[0]}.{int(start.split('.')[1]) + len(match.group(0))}"
                    result_text_widget.tag_add('color_{}'.format(color), start, end)

if __name__ == "__main__":
    app = UIApp()
    app.create_UI()
