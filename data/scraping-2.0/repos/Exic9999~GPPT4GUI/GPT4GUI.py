import tkinter as tk
from tkinter import scrolledtext, font
import openai
import threading

def send_request(event=None):
    def api_call():
        user_input = user_input_box.get("1.0", tk.END).strip()
        clear_output = clear_output_check_var.get()
        if clear_output:
            output_box.configure(state='normal')
            output_box.delete("1.0", tk.END)
            output_box.configure(state='disabled')
        if user_input.lower() == 'exit':
            root.quit()
        else:
            try:
                loading_label.config(text="Asking ChatGPT4 now...")
                response = openai.ChatCompletion.create(
                    model="gpt-4-1106-preview",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=4096
                )
                output_box.configure(state='normal')
                # Insert and apply bold tag to user input
                output_box.insert(tk.END, "You: ")
                start = output_box.index("end-1c linestart")
                output_box.insert(tk.END, user_input + "\n")
                end = output_box.index("end-1c linestart")
                output_box.tag_add("bold", start, end)
                # Insert GPT-4 response
                output_box.insert(tk.END, "GPT-4: " + response['choices'][0]['message']['content'] + "\n\n")
                output_box.configure(state='disabled')
                output_box.yview(tk.END)
                loading_label.config(text="")
            except Exception as e:
                output_box.configure(state='normal')
                output_box.insert(tk.END, "Error: " + str(e) + "\n")
                output_box.configure(state='disabled')
                loading_label.config(text="")

    threading.Thread(target=api_call).start()

openai.api_key = ''

root = tk.Tk()
root.title("GPT-4 GUI")
root.geometry("1500x1000")
root.configure(bg="#f0f0f0")

input_font = font.Font(family="Times New Roman", size=14)
output_font = font.Font(family="Times New Roman", size=14)
bold_font = font.Font(family="Times New Roman", size=14, weight="bold")  # Bold font

input_frame = tk.Frame(root)
input_frame.pack(padx=10, pady=5, fill='both', expand=True)

user_input_box = scrolledtext.ScrolledText(input_frame, height=4, width=70, font=input_font, bg="#7FFFD4")
user_input_box.pack(side='left', fill='both', expand=True)
user_input_box.bind("<Return>", send_request)

send_button = tk.Button(input_frame, text="Send", command=send_request, bg="#4CAF50", fg="white", padx=10, pady=5)
send_button.pack(side='right', padx=10)
send_button.bind("<Enter>", lambda e: e.widget.config(bg="#45a049"))
send_button.bind("<Leave>", lambda e: e.widget.config(bg="#4CAF50"))

loading_label = tk.Label(input_frame, text="", font=("Helvetica", 10))
loading_label.pack(side='right')

clear_output_check_var = tk.BooleanVar()
clear_output_check = tk.Checkbutton(input_frame, text="Clear output on send", var=clear_output_check_var, bg="#f0f0f0")
clear_output_check.pack(side='right')

output_box = scrolledtext.ScrolledText(root, height=15, width=100, font=output_font, bg="#ADD8E6")
output_box.pack(padx=10, pady=5, fill='both', expand=True)
output_box.configure(state='disabled')
output_box.tag_configure("bold", font=bold_font)  # Configure bold tag

root.mainloop()
